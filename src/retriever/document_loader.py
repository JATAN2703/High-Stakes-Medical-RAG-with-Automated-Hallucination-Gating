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

    Falls back to realistic synthetic drug safety documents if the
    DailyMed API is unavailable or returns no data.

    Parameters
    ----------
    data_dir : str | Path
        Local directory to cache downloaded XML files.
    max_labels : int
        Maximum number of drug labels to download.

    Examples
    --------
    >>> loader = DailyMedLoader(data_dir="data/dailymed", max_labels=100)
    >>> docs = loader.load()
    >>> print(len(docs), "document chunks loaded")
    """

    BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

    # SPL section codes for drug safety content
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
        # Check for synthetic cache first
        synthetic_cache = self.data_dir / "synthetic_docs.json"
        if synthetic_cache.exists():
            logger.info("Loading synthetic drug documents from cache.")
            return self._load_synthetic_cache(synthetic_cache)

        # Check for real XML cache
        cached_files = list(self.data_dir.glob("*.xml"))
        if cached_files:
            logger.info(f"Loading {len(cached_files)} cached DailyMed XML files.")
            docs = []
            for xml_path in cached_files[: self.max_labels]:
                docs.extend(self._parse_xml(xml_path))
            logger.info(f"Loaded {len(docs)} document chunks from DailyMed cache.")
            if docs:
                return docs

        logger.info("No cache found. Attempting DailyMed API download...")
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

    def _load_synthetic_cache(self, cache_file: Path) -> list[Document]:
        """Load previously generated synthetic documents from JSON cache."""
        raw = json.loads(cache_file.read_text())
        docs = []
        for item in raw:
            try:
                docs.append(Document(
                    doc_id=item["doc_id"],
                    content=item["content"],
                    source=item["source"],
                    metadata=item.get("metadata", {}),
                ))
            except Exception:
                continue
        logger.info(f"Loaded {len(docs)} synthetic documents from cache.")
        return docs

    def _download_and_parse(self) -> list[Document]:
        """Try DailyMed API endpoints, fall back to synthetic documents."""
        docs = []

        # Strategy 1: spls.json endpoint
        set_ids = self._fetch_set_ids_via_spls()

        if set_ids:
            downloaded = 0
            for set_id in set_ids[: self.max_labels]:
                xml_path = self.data_dir / f"{set_id}.xml"
                if not xml_path.exists():
                    self._download_label(set_id, xml_path)
                    time.sleep(0.3)
                parsed = self._parse_xml(xml_path)
                if parsed:
                    docs.extend(parsed)
                    downloaded += 1
            logger.info(f"Downloaded and parsed {downloaded} drug labels → {len(docs)} chunks.")

        if not docs:
            logger.warning(
                "DailyMed API returned no usable documents. "
                "Using synthetic drug safety corpus for pipeline development."
            )
            return self._generate_synthetic_documents()

        return docs

    def _fetch_set_ids_via_spls(self) -> list[str]:
        """Fetch set IDs from DailyMed SPL listing endpoint."""
        set_ids = []
        for endpoint in [
            f"{self.BASE_URL}/spls.json",
            f"{self.BASE_URL}/drugnames.json",
        ]:
            try:
                resp = requests.get(
                    endpoint,
                    params={"pagesize": min(self.max_labels, 100), "page": 1},
                    timeout=20,
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.debug(f"Response from {endpoint}: keys={list(data.keys())}")

                for key in ["data", "spls", "results", "drugnames"]:
                    items = data.get(key, [])
                    if items:
                        for item in items:
                            sid = (item.get("setid") or item.get("set_id")
                                   or item.get("id") or item.get("spl_id"))
                            if sid:
                                set_ids.append(str(sid))
                        break

                if set_ids:
                    logger.info(f"Fetched {len(set_ids)} set IDs from {endpoint}")
                    return set_ids

            except Exception as e:
                logger.warning(f"Endpoint {endpoint} failed: {e}")

        return set_ids

    def _download_label(self, set_id: str, dest: Path) -> None:
        """Fetch a single SPL XML and save it to disk."""
        url = f"{self.BASE_URL}/spls/{set_id}.xml"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100:
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

                safe_drug = drug_name.lower().replace(" ", "_").replace("/", "_")[:40]
                doc_id = f"{xml_path.stem}_{section_name}_{len(docs):03d}"
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

    def _generate_synthetic_documents(self) -> list[Document]:
        """
        Generate realistic synthetic drug safety documents as a fallback.

        Based on publicly available FDA drug safety information.
        Suitable for pipeline development and hallucination detection benchmarking.
        """
        logger.info("Generating synthetic drug safety corpus...")

        synthetic_data = [
            ("warfarin", "adverse_reactions",
             "Warfarin sodium can cause major or fatal bleeding. Signs and symptoms of bleeding "
             "include headache, dizziness, weakness, prolonged bleeding from cuts, increased menstrual "
             "flow, nausea, vomiting, abdominal pain, and red or dark brown urine. Risk factors for "
             "bleeding include intensity of anticoagulation (INR >4.0), age ≥65, highly variable INR, "
             "history of GI bleeding, hypertension, cerebrovascular disease, and serious heart disease. "
             "Necrosis and gangrene of skin and other tissues can occur. Purple toes syndrome may develop."),
            ("warfarin", "drug_interactions",
             "Warfarin interacts with many drugs. Drugs that increase anticoagulant effect include: "
             "NSAIDs (ibuprofen, naproxen, celecoxib), aspirin, clopidogrel, amiodarone, fluconazole, "
             "metronidazole, ciprofloxacin, clarithromycin, and omeprazole. Drugs that decrease effect: "
             "rifampin, carbamazepine, phenytoin, St. John's Wort, and nafcillin. Monitor INR closely "
             "when initiating or stopping any concomitant drug. Significant interactions also occur with "
             "alcohol, cranberry juice, and grapefruit."),
            ("warfarin", "contraindications",
             "Warfarin is contraindicated in: pregnancy — causes fetal hemorrhage and death; patients "
             "with hemorrhagic tendencies or blood dyscrasias; recent CNS or ophthalmic surgery; "
             "active bleeding from GI, GU, or respiratory tract; threatened abortion; bacterial "
             "endocarditis; hypersensitivity to warfarin or any component of the product."),
            ("warfarin", "warnings",
             "Bleeding risk: warfarin can cause serious and potentially fatal bleeding. Perform regular "
             "INR monitoring. Drugs, dietary changes, and patient factors all affect INR. Necrosis and "
             "gangrene: rare but serious — often occurs 3-10 days after initiation, usually in patients "
             "with protein C or S deficiency. Systemic atheroemboli and cholesterol microemboli: purple "
             "toes syndrome may occur. Limb ischemia, infarction, and gangrene have been reported."),
            ("metformin", "adverse_reactions",
             "Most common adverse reactions (>5%): diarrhea (53.2%), nausea/vomiting (25.5%), "
             "flatulence (12.1%), asthenia (9.2%), indigestion (7.1%), abdominal discomfort (6.4%), "
             "headache (5.7%). GI symptoms most common during initiation and generally transient. "
             "Lactic acidosis: rare (0.03 cases/1000 patient-years) but serious — fatal in ~50% of "
             "cases. Symptoms include malaise, myalgias, respiratory distress, abdominal pain, and "
             "hypothermia. Vitamin B12 deficiency occurs in ~7% with long-term use."),
            ("metformin", "drug_interactions",
             "Carbonic anhydrase inhibitors (topiramate, acetazolamide, zonisamide) increase lactic "
             "acidosis risk. Cationic drugs (amiloride, digoxin, morphine, quinidine, ranitidine, "
             "triamterene, trimethoprim, vancomycin) compete for tubular transport systems and may "
             "increase metformin plasma levels. Iodinated contrast agents may cause acute kidney "
             "injury — withhold metformin before and for 48 hours after contrast administration. "
             "Alcohol potentiates metformin effect on lactate metabolism."),
            ("metformin", "contraindications",
             "Metformin is contraindicated in: eGFR below 30 mL/min/1.73m2 (increased lactic acidosis "
             "risk); acute or chronic metabolic acidosis including diabetic ketoacidosis with or without "
             "coma; hypersensitivity to metformin hydrochloride. Temporarily discontinue in patients "
             "undergoing radiologic procedures with intravascular iodinated contrast. Hold for surgical "
             "procedures requiring restricted food and fluid intake."),
            ("lisinopril", "adverse_reactions",
             "Hypertension trials: hypotension (1.2%), dizziness (6.3%), headache (5.7%), diarrhea "
             "(2.7%), fatigue (3.3%). Dry cough occurs in 2.5-35% of patients per published reports — "
             "a class effect of ACE inhibitors. Angioedema occurs in 0.1-0.5%; Black patients have "
             "higher incidence. Hyperkalemia (serum K+ >5.7 mEq/L) in 2.2-6.0% of patients. "
             "Renal impairment, elevated creatinine (1.8-fold), and rare cases of acute renal failure "
             "and renal artery stenosis exacerbation reported."),
            ("lisinopril", "drug_interactions",
             "NSAIDs including COX-2 inhibitors may reduce antihypertensive effect and worsen renal "
             "function particularly in elderly, volume-depleted, or impaired renal function patients. "
             "Potassium-sparing diuretics (spironolactone, eplerenone, amiloride) and potassium "
             "supplements increase hyperkalemia risk — monitor serum potassium. Lithium: increased "
             "serum lithium levels and toxicity reported. Dual RAAS blockade with ARBs or aliskiren: "
             "increased hypotension, hyperkalemia, renal impairment — avoid combination."),
            ("lisinopril", "contraindications",
             "Contraindicated in: history of ACE inhibitor-associated angioedema; hereditary or "
             "idiopathic angioedema; concomitant use with aliskiren in patients with diabetes; "
             "concomitant use with sacubitril/valsartan within 36 hours of each other; "
             "pregnancy — causes fetal renal dysplasia, oligohydramnios, skull hypoplasia, "
             "pulmonary hypoplasia, limb contractures, and neonatal death. Stop immediately if "
             "pregnancy is detected."),
            ("atorvastatin", "adverse_reactions",
             "Clinical adverse reactions (≥2% and greater than placebo): nasopharyngitis (8.3%), "
             "arthralgia (6.9%), diarrhea (6.8%), pain in extremity (6.0%), UTI (5.7%), dyspepsia "
             "(4.7%), nausea (4.0%), musculoskeletal pain (3.8%), muscle spasms (3.6%), insomnia (3.0%). "
             "Myopathy/rhabdomyolysis: rare (<0.1%). Serum transaminase increases >3x ULN: 0.7%. "
             "Hemorrhagic stroke: increased risk in patients with recent stroke or TIA. "
             "Immune-mediated necrotizing myopathy reported rarely."),
            ("atorvastatin", "drug_interactions",
             "Atorvastatin metabolized by CYP3A4. Strong CYP3A4 inhibitors — clarithromycin, "
             "itraconazole, ketoconazole, HIV protease inhibitors (lopinavir, ritonavir), hepatitis C "
             "protease inhibitors, nefazodone — markedly increase atorvastatin exposure and myopathy "
             "risk. Cyclosporine increases AUC 8.7-fold — limit atorvastatin to 10mg/day. Gemfibrozil "
             "increases myopathy risk — avoid combination. Colchicine combined with statins: myopathy "
             "cases reported. Diltiazem and verapamil increase atorvastatin AUC ~2.5-3.4 fold."),
            ("aspirin", "adverse_reactions",
             "GI reactions: dyspepsia, nausea, vomiting, gross GI bleeding, peptic ulcers, and "
             "gastritis. GI bleeding can be serious. CNS effects: agitation, cerebral edema, coma, "
             "confusion, dizziness, headache, lethargy, seizures. Hearing: tinnitus and reversible "
             "hearing loss at high doses (>3g/day). Hematologic: prolonged bleeding time, "
             "thrombocytopenia, and disseminated intravascular coagulation. Hypersensitivity: "
             "urticaria, angioedema, bronchospasm (0.2-0.9%). Reye's syndrome in children."),
            ("aspirin", "drug_interactions",
             "Anticoagulants: aspirin increases bleeding risk with warfarin, heparin, dabigatran. "
             "NSAIDs: may antagonize irreversible platelet inhibition by aspirin; use ibuprofen "
             "at least 30 minutes after or 8 hours before aspirin if both required. ACE inhibitors: "
             "high-dose aspirin may reduce antihypertensive effect. Antidiabetic agents: salicylates "
             "may enhance hypoglycemic effect. Methotrexate: aspirin may increase methotrexate toxicity "
             "by displacement from plasma proteins and reduced renal clearance."),
            ("amoxicillin", "adverse_reactions",
             "GI: diarrhea, gastritis, nausea, vomiting, hemorrhagic colitis, Clostridioides "
             "difficile-associated diarrhea (CDAD — ranging from mild diarrhea to fatal colitis). "
             "Hypersensitivity: rash, urticaria, serum sickness-like reactions, erythema multiforme, "
             "Stevens-Johnson syndrome, anaphylaxis — serious and occasionally fatal anaphylaxis "
             "reported. Maculopapular rash occurs frequently in patients with EBV mononucleosis. "
             "CNS: agitation, anxiety, confusion, convulsions (high doses), dizziness."),
            ("amoxicillin", "drug_interactions",
             "Probenecid: blocks renal tubular secretion of amoxicillin, increasing plasma levels — "
             "do not use to extend amoxicillin half-life. Anticoagulants: abnormal prolongation of "
             "prothrombin time with amoxicillin; monitor PT/INR in patients on anticoagulants. "
             "Oral contraceptives: may reduce effectiveness due to gut flora disruption. "
             "Allopurinol: increases incidence of rashes. Bacteriostatic antibiotics — "
             "chloramphenicol, tetracyclines, sulfonamides — may antagonize bactericidal effect."),
            ("metoprolol", "adverse_reactions",
             "CNS: tiredness (10%), dizziness (10%), depression (5%), headache (6%). Cardiovascular: "
             "bradycardia (3-4% dose-related), heart failure worsening, hypotension. Respiratory: "
             "shortness of breath (3%), bronchospasm in susceptible patients. GI: diarrhea (5%), "
             "nausea (3.4%), dry mouth (1.4%). Abrupt withdrawal may exacerbate angina, MI, and "
             "ventricular arrhythmia — taper over 1-2 weeks when discontinuing."),
            ("metoprolol", "drug_interactions",
             "Catecholamine-depleting drugs (reserpine, MAOIs): may produce excessive reduction in "
             "sympathetic activity — monitor for hypotension and bradycardia. CYP2D6 inhibitors "
             "(fluoxetine, paroxetine, propafenone, quinidine): increase metoprolol plasma levels "
             "up to 5-fold. Clonidine: if withdrawn while on beta-blocker, rebound hypertension "
             "may occur — taper clonidine slowly. Calcium channel blockers (verapamil, diltiazem): "
             "additive negative chronotropic and inotropic effects."),
        ]

        docs = []
        for drug_name, section, content in synthetic_data:
            doc_id = f"synthetic_{drug_name}_{section}"
            docs.append(Document(
                doc_id=doc_id,
                content=content,
                source=drug_name.title(),
                metadata={
                    "drug_name": drug_name,
                    "section": section,
                    "source_type": "synthetic",
                    "set_id": doc_id,
                }
            ))

        logger.info(f"Generated {len(docs)} synthetic drug safety documents.")

        # Cache for subsequent runs
        cache_file = self.data_dir / "synthetic_docs.json"
        cache_data = [
            {"doc_id": d.doc_id, "content": d.content,
             "source": d.source, "metadata": d.metadata}
            for d in docs
        ]
        cache_file.write_text(json.dumps(cache_data, indent=2))
        logger.info(f"Synthetic corpus cached to {cache_file}")
        return docs


# ── FAERS Loader ───────────────────────────────────────────────────────────────

class FAERSLoader:
    """
    Downloads adverse event reports from the FDA FAERS public API.

    Parameters
    ----------
    data_dir : str | Path
        Local directory to cache downloaded FAERS data.
    max_reports : int
        Maximum number of adverse event reports to fetch.
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
        reactions = report.get("patient", {}).get("reaction", [])
        drugs = report.get("patient", {}).get("drug", [])

        drug_names = [d.get("medicinalproduct", "") for d in drugs if d.get("medicinalproduct")]
        reaction_names = [r.get("reactionmeddrapt", "") for r in reactions]

        if drug_names and reaction_names:
            return (
                f"Patient received {', '.join(drug_names)}. "
                f"Reported adverse reactions: {', '.join(reaction_names)}."
            )
        return ""