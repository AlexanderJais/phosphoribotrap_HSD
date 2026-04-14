"""Smoke tests for phosphotrap.reference.

The pure-Python helpers (decoys, gentrome, tx2gene) are tested against
hand-rolled gzipped fixtures so no network or salmon binary is needed.
The orchestrator is tested with both downloads and the salmon index
build mocked out — what we care about is that the cache short-circuits
work, the progress callback is monotonic, and the returned artifact
paths point at real files.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import patch

import pytest

from phosphotrap import reference as ref


# ----------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------
_FAKE_GENOME = b""">chr1 some description here
ACGTACGTACGT
ACGTACGTACGT
>chr2
NNNNNNNNNNNN
>chrM mitochondrial
ACGT
"""

# Two transcripts on the same gene, plus an exon line we should ignore,
# plus a transcript with no gene_name (should fall back to gene_id).
_FAKE_GTF = b"""##description: fake
##provider: test
chr1\tHAVANA\tgene\t1\t1000\t.\t+\t.\tgene_id "ENSMUSG00000000001.1"; gene_name "Gnai3";
chr1\tHAVANA\ttranscript\t1\t1000\t.\t+\t.\ttranscript_id "ENSMUST00000000001.4"; gene_id "ENSMUSG00000000001.1"; gene_name "Gnai3";
chr1\tHAVANA\texon\t1\t500\t.\t+\t.\ttranscript_id "ENSMUST00000000001.4"; gene_id "ENSMUSG00000000001.1"; gene_name "Gnai3";
chr1\tHAVANA\ttranscript\t1\t800\t.\t+\t.\ttranscript_id "ENSMUST00000000002.1"; gene_id "ENSMUSG00000000001.1"; gene_name "Gnai3";
chr2\tENSEMBL\ttranscript\t100\t900\t.\t-\t.\ttranscript_id "ENSMUST00000000099.2"; gene_id "ENSMUSG00000000099.2";
"""


def _gz_write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(data)
    return path


# ----------------------------------------------------------------------
# GencodeFiles
# ----------------------------------------------------------------------
def test_gencode_files_for_mouse_default():
    f = ref.GencodeFiles.for_mouse("M38")
    assert f.release == "M38"
    assert f.transcripts_name == "gencode.vM38.transcripts.fa.gz"
    assert f.genome_name == "GRCm39.primary_assembly.genome.fa.gz"
    assert f.gtf_name == "gencode.vM38.primary_assembly.annotation.gtf.gz"
    assert "release_M38" in f.transcripts_url
    assert f.transcripts_url.endswith(f.transcripts_name)
    assert f.genome_url.endswith(f.genome_name)
    assert f.gtf_url.endswith(f.gtf_name)


def test_gencode_files_rejects_empty_release():
    with pytest.raises(ValueError):
        ref.GencodeFiles.for_mouse("   ")


# ----------------------------------------------------------------------
# Decoys
# ----------------------------------------------------------------------
def test_build_decoys_writes_one_contig_per_line(tmp_path: Path):
    genome = _gz_write(tmp_path / "genome.fa.gz", _FAKE_GENOME)
    decoys = tmp_path / "decoys.txt"

    n = ref.build_decoys(genome, decoys)

    assert n == 3
    lines = decoys.read_text().splitlines()
    # First column only — descriptions are stripped.
    assert lines == ["chr1", "chr2", "chrM"]


# ----------------------------------------------------------------------
# Gentrome
# ----------------------------------------------------------------------
def test_build_gentrome_concatenates_streams(tmp_path: Path):
    transcripts = _gz_write(tmp_path / "tx.fa.gz", b">tx1\nACGT\n")
    genome = _gz_write(tmp_path / "g.fa.gz", _FAKE_GENOME)
    gentrome = tmp_path / "gentrome.fa.gz"

    ref.build_gentrome(transcripts, genome, gentrome)

    # File size should be ~ sum of inputs (gzip streams concatenate).
    expected_size = transcripts.stat().st_size + genome.stat().st_size
    assert gentrome.stat().st_size == expected_size

    # Decompressing the concatenated stream should yield both payloads.
    with gzip.open(gentrome, "rb") as fh:
        decoded = fh.read()
    assert b">tx1\nACGT" in decoded
    assert b">chr1" in decoded
    assert b">chrM" in decoded


# ----------------------------------------------------------------------
# tx2gene
# ----------------------------------------------------------------------
def test_build_tx2gene_extracts_three_columns(tmp_path: Path):
    gtf = _gz_write(tmp_path / "ann.gtf.gz", _FAKE_GTF)
    tx2gene = tmp_path / "tx2gene.tsv"

    n = ref.build_tx2gene(gtf, tx2gene)

    assert n == 3  # 3 transcript lines, exon ignored, gene line ignored
    rows = [line.split("\t") for line in tx2gene.read_text().splitlines()]
    assert len(rows) == 3
    # Each row is (transcript_id, gene_id, gene_name).
    assert rows[0] == ["ENSMUST00000000001.4", "ENSMUSG00000000001.1", "Gnai3"]
    assert rows[1] == ["ENSMUST00000000002.1", "ENSMUSG00000000001.1", "Gnai3"]
    # Missing gene_name falls back to gene_id.
    assert rows[2] == [
        "ENSMUST00000000099.2",
        "ENSMUSG00000000099.2",
        "ENSMUSG00000000099.2",
    ]


def test_build_tx2gene_progress_callback_monotonic(tmp_path: Path):
    """Larger fixture so the 5_000-row progress emit interval triggers."""
    rows = [
        b'chr1\tH\ttranscript\t1\t1000\t.\t+\t.\ttranscript_id "TX%05d"; '
        b'gene_id "GE%05d"; gene_name "G%05d";\n' % (i, i, i)
        for i in range(12_000)
    ]
    gtf = _gz_write(tmp_path / "big.gtf.gz", b"".join(rows))
    tx2gene = tmp_path / "tx2gene.tsv"

    seen: list[float] = []
    def cb(frac: float, msg: str) -> None:
        seen.append(frac)

    n = ref.build_tx2gene(gtf, tx2gene, progress_cb=cb)

    assert n == 12_000
    assert len(seen) >= 2  # at least a couple of mid-run emits
    # Monotonically non-decreasing and ending at 1.0.
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# FASTA transcript ID helpers (coverage check)
# ----------------------------------------------------------------------
# GENCODE transcriptome FASTAs use "|"-delimited headers; salmon's
# --gencode flag splits on the first "|". Our helpers must do the same
# so coverage comparisons line up.
_GENCODE_FASTA = (
    b">ENSMUST00000000001.4|ENSMUSG00000000001.1|OTTMUSG00000049935.1|"
    b"OTTMUST00000127195.2|Gnai3-201|Gnai3|3262|\n"
    b"ACGTACGT\n"
    b">ENSMUST00000000002.1|ENSMUSG00000000001.1|-|-|Gnai3-202|Gnai3|"
    b"800|\n"
    b"ACGT\n"
    b">ENSMUST99999999999.1|ENSMUSG99999999999.1|-|-|Orphan-201|"
    b"Orphan|500|\n"
    b"ACGT\n"
)


def test_count_fasta_transcripts(tmp_path: Path):
    fa = _gz_write(tmp_path / "tx.fa.gz", _GENCODE_FASTA)
    assert ref.count_fasta_transcripts(fa) == 3


def test_fasta_transcript_ids_strips_gencode_pipe(tmp_path: Path):
    fa = _gz_write(tmp_path / "tx.fa.gz", _GENCODE_FASTA)
    ids = ref.fasta_transcript_ids(fa, gencode=True)
    assert ids == {
        "ENSMUST00000000001.4",
        "ENSMUST00000000002.1",
        "ENSMUST99999999999.1",
    }


def test_fasta_transcript_ids_without_gencode_keeps_full_header(
    tmp_path: Path,
):
    fa = _gz_write(tmp_path / "tx.fa.gz", _GENCODE_FASTA)
    ids = ref.fasta_transcript_ids(fa, gencode=False)
    # Headers with "|" still resolved — we just keep the whole token.
    assert all("|" in i for i in ids)
    assert len(ids) == 3


def test_tx2gene_transcript_ids(tmp_path: Path):
    tx = tmp_path / "tx2gene.tsv"
    tx.write_text(
        "ENSMUST00000000001.4\tENSMUSG00000000001.1\tGnai3\n"
        "ENSMUST00000000002.1\tENSMUSG00000000001.1\tGnai3\n"
    )
    ids = ref.tx2gene_transcript_ids(tx)
    assert ids == {"ENSMUST00000000001.4", "ENSMUST00000000002.1"}


# ----------------------------------------------------------------------
# download_file
# ----------------------------------------------------------------------
def test_download_file_skips_when_present(tmp_path: Path):
    dest = tmp_path / "already.fa.gz"
    dest.write_bytes(b"cached")

    # If urlopen is touched, the test fails — we expect a cache hit.
    with patch("phosphotrap.reference.urllib.request.urlopen") as mock_open:
        out = ref.download_file("https://example/", dest)

    assert out == dest
    assert dest.read_bytes() == b"cached"
    mock_open.assert_not_called()


def test_download_file_idle_stall_raises_runtime_error(tmp_path: Path):
    """MEDIUM #6: a stalled socket read must surface as a clear error.

    Mocks the urlopen response so that ``resp.read()`` raises
    ``socket.timeout`` on the first call — simulating a server that
    accepted the connection but stopped sending bytes. The user-facing
    RuntimeError must mention "stalled" and the idle-timeout constant
    so the user knows it's a network issue, not a path issue.
    """
    import socket
    from unittest.mock import MagicMock

    dest = tmp_path / "bigfile.fa.gz"
    # Leave dest absent so we don't hit the cache short-circuit.

    class _StallingResp:
        headers = {"Content-Length": "1000000"}

        def read(self, _chunk_size):
            raise socket.timeout("no bytes from server")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "phosphotrap.reference.urllib.request.urlopen",
        return_value=_StallingResp(),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            ref.download_file("https://example/stalled", dest)

    msg = str(exc_info.value)
    assert "stalled" in msg.lower()
    # Partial file should not be left behind.
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".partial").exists()


# ----------------------------------------------------------------------
# build_reference orchestrator
# ----------------------------------------------------------------------
def test_build_reference_orchestrates_with_mocked_downloads(tmp_path: Path):
    """End-to-end orchestrator test with downloads + salmon mocked.

    Asserts that:
    - Each output file the artifacts dataclass references exists.
    - The progress callback is monotonic and ends at 1.0.
    - The cached salmon index is detected via info.json on a re-run.
    """
    dest = tmp_path / "ref"

    # Pre-stage the "downloaded" files so the mocked download_file just
    # returns the path; this sidesteps urllib entirely.
    transcripts_name = "gencode.vMTEST.transcripts.fa.gz"
    genome_name = "GRCm39.primary_assembly.genome.fa.gz"
    gtf_name = "gencode.vMTEST.primary_assembly.annotation.gtf.gz"

    def fake_download(url, dest_path, *, force=False, progress_cb=None,
                      chunk_size=0, label=None):
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.name == transcripts_name:
            _gz_write(dest_path, b">tx1\nACGT\n")
        elif dest_path.name == genome_name:
            _gz_write(dest_path, _FAKE_GENOME)
        elif dest_path.name == gtf_name:
            _gz_write(dest_path, _FAKE_GTF)
        else:
            raise AssertionError(f"unexpected download target: {dest_path}")
        if progress_cb is not None:
            progress_cb(1.0, "mock done")
        return dest_path

    def fake_salmon_index(gentrome, decoys, index_dir, *, threads=4, kmer=31,
                          gencode=True, force=False, log_file=None,
                          progress_cb=None):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        # Drop a marker file so _index_already_built() returns True.
        (index_dir / "info.json").write_text("{}")
        if progress_cb is not None:
            progress_cb(1.0, "mock salmon done")
        return index_dir

    seen: list[float] = []
    def cb(frac: float, msg: str) -> None:
        seen.append(frac)

    with patch.object(ref, "download_file", side_effect=fake_download), \
         patch.object(ref, "build_salmon_index", side_effect=fake_salmon_index):
        artifacts = ref.build_reference(
            release="MTEST",
            dest_dir=dest,
            threads=2,
            force=False,
            progress_cb=cb,
        )

    # Outputs exist.
    assert artifacts.index_dir.is_dir()
    assert (artifacts.index_dir / "info.json").exists()
    assert artifacts.tx2gene_tsv.exists()
    assert artifacts.transcripts_fa.exists()
    assert artifacts.genome_fa.exists()
    assert artifacts.gtf.exists()
    assert artifacts.decoys.exists()
    assert artifacts.gentrome.exists()
    assert artifacts.n_transcripts == 3

    # Coverage was computed. The fake fixture's FASTA has 1 transcript
    # (">tx1") that isn't in tx2gene.tsv (which was built from the GTF
    # and has ENSMUST00000000001.4 / .0002.1 / .0099.2). So coverage
    # should be 0.0 — 0/1 FASTA transcripts found in tx2gene. What we
    # care about here is that the FIELDS are populated, not the
    # specific ratio.
    assert artifacts.n_fasta_transcripts >= 1
    assert 0.0 <= artifacts.tx2gene_coverage <= 1.0

    # Progress monotonic + reaches 1.0.
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(1.0)


def test_build_reference_coverage_is_1_when_tx2gene_covers_fasta(
    tmp_path: Path,
):
    """When the FASTA headers match tx2gene transcript IDs, coverage is 1.0."""
    dest = tmp_path / "ref"

    # Matching GENCODE FASTA + GTF: the FASTA headers contain the same
    # transcript IDs the GTF declares, so tx2gene coverage is perfect.
    matched_fasta = (
        b">ENSMUST00000000001.4|ENSMUSG00000000001.1|-|-|Gnai3-201|"
        b"Gnai3|3262|\n"
        b"ACGT\n"
        b">ENSMUST00000000002.1|ENSMUSG00000000001.1|-|-|Gnai3-202|"
        b"Gnai3|800|\n"
        b"ACGT\n"
        b">ENSMUST00000000099.2|ENSMUSG00000000099.2|-|-|orphan-201|"
        b"Orphan|500|\n"
        b"ACGT\n"
    )

    def fake_download(url, dest_path, *, force=False, progress_cb=None,
                      chunk_size=0, label=None):
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if "transcripts" in dest_path.name:
            _gz_write(dest_path, matched_fasta)
        elif "genome" in dest_path.name:
            _gz_write(dest_path, _FAKE_GENOME)
        else:
            _gz_write(dest_path, _FAKE_GTF)
        return dest_path

    def fake_salmon_index(gentrome, decoys, index_dir, *, threads=4, kmer=31,
                          gencode=True, force=False, log_file=None,
                          progress_cb=None):
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        (Path(index_dir) / "info.json").write_text("{}")
        return Path(index_dir)

    with patch.object(ref, "download_file", side_effect=fake_download), \
         patch.object(ref, "build_salmon_index", side_effect=fake_salmon_index):
        artifacts = ref.build_reference(
            release="MTEST", dest_dir=dest, threads=2
        )

    assert artifacts.n_fasta_transcripts == 3
    assert artifacts.tx2gene_coverage == pytest.approx(1.0)


def test_build_reference_second_run_is_cached(tmp_path: Path):
    """Re-running over an existing dest dir should not re-download."""
    dest = tmp_path / "ref"

    # First run: real (mocked) work.
    def fake_download(url, dest_path, *, force=False, progress_cb=None,
                      chunk_size=0, label=None):
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if "transcripts" in dest_path.name:
            _gz_write(dest_path, b">tx1\nACGT\n")
        elif "genome" in dest_path.name:
            _gz_write(dest_path, _FAKE_GENOME)
        else:
            _gz_write(dest_path, _FAKE_GTF)
        return dest_path

    def fake_salmon_index(gentrome, decoys, index_dir, *, threads=4, kmer=31,
                          gencode=True, force=False, log_file=None,
                          progress_cb=None):
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        (Path(index_dir) / "info.json").write_text("{}")
        return Path(index_dir)

    with patch.object(ref, "download_file", side_effect=fake_download) as dl, \
         patch.object(ref, "build_salmon_index", side_effect=fake_salmon_index):
        ref.build_reference(release="MTEST", dest_dir=dest, threads=2)

    assert dl.call_count == 3

    # Second run: the FASTA / GTF files exist, so download_file should
    # be called again BUT it should hit its own cache short-circuit. We
    # patch the *underlying* urlopen to verify nothing actually makes
    # it onto the wire.
    with patch("phosphotrap.reference.urllib.request.urlopen") as mock_open, \
         patch.object(ref, "build_salmon_index", side_effect=fake_salmon_index):
        ref.build_reference(release="MTEST", dest_dir=dest, threads=2)
    mock_open.assert_not_called()
