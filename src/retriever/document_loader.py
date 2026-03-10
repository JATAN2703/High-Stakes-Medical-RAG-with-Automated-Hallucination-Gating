"""
src/retriever/document_loader.py
=================================
Loaders for FDA DailyMed drug labels (XML) and FAERS adverse event reports.

Each loader returns a list of ``Document`` objects — a lightweight dataclass
holding text content and metadata. All parsing logic is isolated here so
the rest of the pipeline is agnostic to data source format.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import requests
from lxml import etree

from src.utils import get_logger

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """
    A single retrievable unit of text with associated metadata.

    Attributes
    ----------
    doc_id : str
        Unique identifier for this document.
    content : str
        The raw text content of the document chunk.
    source : str
        Human-readable source label (e.g. drug name or report ID).
    metadata : dict
        Arbitrary key-value pairs (drug name, section, date, etc.)
    is_adversarial : bool
        True if this document was injected as part of an adversarial test.
    """
    doc_id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)
    is_adversarial: bool = False

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError(f"Document {self.doc_id} has empty content.")


# ── DailyMed Loader ───────────────────────────────────────────────────────────

class DailyMedLoader:
    """
    Downloads and parses FDA DailyMed drug label XML files.

    DailyMed provides structured SPL (Structured Product Labeling) XML
    for every FDA-approved drug. This loader extracts the clinically
    relevant sections (adverse reactions, drug interactions, warnings)
    and splits them into Document chunks.

    Parameters
    ----------
    data_dir : str | Path
        Local directory to cache downloaded XML files.
    max_labels : int
        Maximum number of drug labels to download. Use a small number
        (e.g. 50) during development.

    Examples
    --------
    >>> loader = DailyMedLoader(data_dir="data/dailymed", max_labels=100)
    >>> docs = loader.load()
    >>> print(len(docs), "document chunks loaded")
    """

    BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

    # SPL section codes we care about for drug safety
    TARGET_SECTIONS = {
        "34084-4": "adverse_reactions",
        "34073-7": "drug_interactions",
        "34071-1": "warnings",
        "34068-7": "dosage_and_administration",
        "34070-3": "contraindications",
    }

    def __init__(self, data_dir: str | Path = "data/dailymed", max_labels: int = 500) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_labels = max_labels

    def load(self) -> list[Document]:
        """
        Load documents from local cache or download from DailyMed.

        Returns
        -------
        list[Document]
            Parsed and chunked document list.
        """
        cached_files = list(self.data_dir.glob("*.xml"))
        if cached_files:
            logger.info(f"Loading {len(cached_files)} cached DailyMed XML files.")
            docs = []
            for xml_path in cached_files[: self.max_labels]:
                docs.extend(self._parse_xml(xml_path))
            logger.info(f"Loaded {len(docs)} document chunks from DailyMed cache.")
            return docs

        logger.info("No cache found. Downloading from DailyMed API...")
        return self._download_and_parse()

    def load_from_directory(self, directory: str | Path) -> list[Document]:
        """
        Parse all XML files in a given directory.

        Parameters
        ----------
        directory : str | Path
            Path to folder containing .xml SPL files.

        Returns
        -------
        list[Document]
        """
        directory = Path(directory)
        docs = []
        xml_files = list(directory.glob("*.xml"))
        logger.info(f"Parsing {len(xml_files)} XML files from {directory}")
        for xml_path in xml_files:
            docs.extend(self._parse_xml(xml_path))
        return docs

    def _download_and_parse(self) -> list[Document]:
        """Download drug label index and fetch individual SPL XMLs."""
        docs = []
        page = 1
        downloaded = 0

        while downloaded < self.max_labels:
            response = requests.get(
                f"{self.BASE_URL}/drugnames.json",
                params={"pagesize": 100, "page": page},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            drug_list = data.get("data", [])
            if not drug_list:
                break

            for drug in drug_list:
                if downloaded >= self.max_labels:
                    break
                set_id = drug.get("setid")
                if not set_id:
                    continue
                xml_path = self.data_dir / f"{set_id}.xml"
                if not xml_path.exists():
                    self._download_label(set_id, xml_path)
                    time.sleep(0.2)  # be polite to the API
                docs.extend(self._parse_xml(xml_path))
                downloaded += 1

            page += 1

        logger.info(f"Downloaded and parsed {downloaded} drug labels → {len(docs)} chunks.")
        return docs

    def _download_label(self, set_id: str, dest: Path) -> None:
        """Fetch a single SPL XML and save it to disk."""
        url = f"{self.BASE_URL}/spls/{set_id}.xml"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except requests.RequestException as e:
            logger.warning(f"Failed to download {set_id}: {e}")

    def _parse_xml(self, xml_path: Path) -> list[Document]:
        """
        Parse a single SPL XML file into Document chunks.

        Parameters
        ----------
        xml_path : Path
            Path to the SPL XML file.

        Returns
        -------
        list[Document]
            One Document per target section found in the label.
        """
        try:
            tree = etree.parse(str(xml_path))
            root = tree.getroot()
            ns = {"spl": "urn:hl7-org:v3"}

            drug_name = self._extract_drug_name(root, ns)
            docs = []

            for section in root.findall(".//spl:section", ns):
                code_elem = section.find("spl:code", ns)
                if code_elem is None:
                    continue
                code = code_elem.get("code", "")
                section_name = self.TARGET_SECTIONS.get(code)
                if not section_name:
                    continue

                text = self._extract_text(section, ns)
                if len(text.strip()) < 50:
                    continue

                doc_id = f"{xml_path.stem}_{section_name}"
                docs.append(Document(
                    doc_id=doc_id,
                    content=text.strip(),
                    source=drug_name,
                    metadata={
                        "drug_name": drug_name,
                        "section": section_name,
                        "set_id": xml_path.stem,
                        "source_file": str(xml_path),
                    }
                ))

            return docs

        except Exception as e:
            logger.warning(f"Failed to parse {xml_path}: {e}")
            return []

    @staticmethod
    def _extract_drug_name(root: etree._Element, ns: dict) -> str:
        """Pull drug name from the SPL header."""
        name_elem = root.find(".//spl:manufacturedProduct/spl:name", ns)
        if name_elem is not None and name_elem.text:
            return name_elem.text.strip()
        return "Unknown Drug"

    @staticmethod
    def _extract_text(section: etree._Element, ns: dict) -> str:
        """Extract all text content from a section, stripping XML tags."""
        texts = []
        for elem in section.iter():
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                texts.append(elem.tail.strip())
        return " ".join(texts)


# ── FAERS Loader ───────────────────────────────────────────────────────────────

class FAERSLoader:
    """
    Downloads adverse event reports from the FDA FAERS public API.

    FAERS reports are real-world post-market safety data submitted by
    clinicians and patients. They often contain adverse event descriptions
    that differ from official drug labels — a natural source of conflict
    used in adversarial injection experiments.

    Parameters
    ----------
    data_dir : str | Path
        Local directory to cache downloaded FAERS data.
    max_reports : int
        Maximum number of adverse event reports to fetch.

    Examples
    --------
    >>> loader = FAERSLoader(data_dir="data/faers", max_reports=200)
    >>> docs = loader.load(drug_name="warfarin")
    """

    FAERS_URL = "https://api.fda.gov/drug/event.json"

    def __init__(self, data_dir: str | Path = "data/faers", max_reports: int = 500) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_reports = max_reports

    def load(self, drug_name: str | None = None) -> list[Document]:
        """
        Load FAERS reports from cache or download from the API.

        Parameters
        ----------
        drug_name : str | None
            If given, filter reports to this drug name.

        Returns
        -------
        list[Document]
        """
        cache_key = drug_name.lower().replace(" ", "_") if drug_name else "all"
        cache_file = self.data_dir / f"faers_{cache_key}.json"

        if cache_file.exists():
            logger.info(f"Loading FAERS cache: {cache_file}")
            raw = json.loads(cache_file.read_text())
            return self._parse_reports(raw, drug_name)

        logger.info(f"Fetching FAERS reports for: {drug_name or 'all drugs'}")
        raw = self._fetch_reports(drug_name)
        cache_file.write_text(json.dumps(raw, indent=2))
        return self._parse_reports(raw, drug_name)

    def _fetch_reports(self, drug_name: str | None) -> list[dict]:
        """Download adverse event reports from the FDA API."""
        reports = []
        limit = 100
        skip = 0

        while len(reports) < self.max_reports:
            params: dict = {"limit": limit, "skip": skip}
            if drug_name:
                params["search"] = f'patient.drug.medicinalproduct:"{drug_name}"'

            try:
                resp = requests.get(self.FAERS_URL, params=params, timeout=30)
                resp.raise_for_status()
                batch = resp.json().get("results", [])
                if not batch:
                    break
                reports.extend(batch)
                skip += limit
                time.sleep(0.3)
            except requests.RequestException as e:
                logger.warning(f"FAERS API error: {e}")
                break

        return reports[: self.max_reports]

    def _parse_reports(self, reports: list[dict], drug_name: str | None) -> list[Document]:
        """Convert raw FAERS JSON records into Document objects."""
        docs = []
        for i, report in enumerate(reports):
            narrative = self._extract_narrative(report)
            if not narrative or len(narrative) < 30:
                continue

            reactions = [
                r.get("reactionmeddrapt", "")
                for r in report.get("patient", {}).get("reaction", [])
            ]

            docs.append(Document(
                doc_id=f"faers_{report.get('safetyreportid', i)}",
                content=narrative,
                source=f"FAERS Report {report.get('safetyreportid', i)}",
                metadata={
                    "report_id": report.get("safetyreportid"),
                    "drug_name": drug_name or "multiple",
                    "reactions": reactions,
                    "report_date": report.get("receiptdate"),
                    "source_type": "faers",
                }
            ))

        logger.info(f"Parsed {len(docs)} FAERS documents.")
        return docs

    @staticmethod
    def _extract_narrative(report: dict) -> str:
        """Pull the case narrative or construct one from reaction data."""
        narrative = report.get("companynumb", "")
        reactions = report.get("patient", {}).get("reaction", [])
        drugs = report.get("patient", {}).get("drug", [])

        drug_names = [d.get("medicinalproduct", "") for d in drugs if d.get("medicinalproduct")]
        reaction_names = [r.get("reactionmeddrapt", "") for r in reactions]

        if drug_names and reaction_names:
            return (
                f"Patient received {', '.join(drug_names)}. "
                f"Reported adverse reactions: {', '.join(reaction_names)}."
            )
        return narrative
