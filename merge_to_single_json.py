
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zip_merge_to_single_json_v9.py
- Restores/ensures these fields when available:
  * registration_date (coalesced across many variants, formatted dd-mm-YYYY if parseable)
  * industrial_classification (falls back to first industry domain label if text missing)
  * company_scale (from raw/label or via CompanyScale master FK)
  * organization_type (from raw/label or via OrganisationType/OrganizationType master FK)
- Keeps v8 improvements: NULL cleanup, label backfills, address inference, whitespace fixes
"""

import argparse, json, zipfile, io, re
from pathlib import Path
import pandas as pd

COMMON_ENCODINGS = ["utf-8","utf-8-sig","latin-1","cp1252","iso-8859-1","utf-16","utf-16le","utf-16be"]
NA_MARKERS = ["", " ", "NULL", "null", "NaN", "nan", "None", None]

def clean_str(v):
    if v is None: 
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s in ("NULL","null","NaN","nan","None"): return ""
    return s

def strip_spaces(s: str) -> str:
    s = clean_str(s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def id_to_str(v: object) -> str:
    s = clean_str(v)
    m = re.fullmatch(r"(\d+)(?:\.0+)?", s)  # 123.0 -> "123"
    return m.group(1) if m else s

def norm_table_name(name: str) -> str:
    if not name: return ""
    n = name.strip()
    if n.lower().endswith(".csv"): n = n[:-4]
    if n.lower().startswith("dbo."): n = n[4:]
    return n

def read_csv_from_zip(zf: zipfile.ZipFile, member: zipfile.ZipInfo):
    for enc in COMMON_ENCODINGS:
        try:
            with zf.open(member, "r") as fh:
                return pd.read_csv(fh, dtype=str, keep_default_na=True, na_values=NA_MARKERS, encoding=enc)
        except Exception:
            continue
    data = zf.read(member)
    for enc in COMMON_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=True, na_values=NA_MARKERS, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read {member.filename}")

def pick_label_column(df: pd.DataFrame):
    if df is None or df.empty: return None
    preferred = [c for c in df.columns if c.lower().endswith("name")]
    if preferred: return preferred[0]
    for guess in ["TypeName","CategoryName","SubCategoryName","Year","Description","Desc","Label","Title","Expertise","CoreExpertise","ScaleName","OrganisationType"]:
        if guess in df.columns: return guess
    for c in df.columns:
        if c.lower() not in ("id","pk","key"):
            return c
    return df.columns[0]

def make_lookup(df: pd.DataFrame, id_col="Id", label_col=None):
    out = {}
    if df is None or df.empty: return out

    # try to find an id-like column if "Id" missing
    if id_col not in df.columns:
        for c in df.columns:
            if c.lower() in ("id","pk","key"):
                id_col = c
                break

    if label_col is None:
        label_col = pick_label_column(df)

    for _, r in df.iterrows():
        rid = id_to_str(r.get(id_col, ""))
        if not rid:
            continue
        d = {c: clean_str(r.get(c, "")) for c in df.columns}
        if label_col and label_col in df.columns:
            d["_label"] = clean_str(r.get(label_col, ""))
        out[rid] = d
    return out

def index_by_company(df: pd.DataFrame, fk_cols=("CompanyMaster_FK_ID","CompanyMaster_Fk_Id","Company_FK_Id","CompanyRefNo","CompanyId","CompanyID","Company_Id")):
    ix = {}
    if df is None or df.empty: return ix
    for _, r in df.iterrows():
        key = None
        for fk in fk_cols:
            if fk in df.columns:
                val = clean_str(r.get(fk, ""))
                if val:
                    key = val
                    break
        if key is None:
            continue
        ix.setdefault(key, []).append({c: clean_str(r.get(c, "")) for c in df.columns})
    return ix

def first_present(row: dict, keys, default=""):
    for k in keys:
        v = row.get(k, "")
        if clean_str(v):
            return clean_str(v)
    return default

# --- address inference (India-centric) ---
INDIA_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana",
    "Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar Islands",
    "Chandigarh","Dadra and Nagar Haveli and Daman and Diu","Delhi","Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"
]
INDIA_STATES_L = [s.lower() for s in INDIA_STATES]
PIN_RE = re.compile(r"\b(\d{6})\b")
def infer_from_address(addr: str):
    s = clean_str(addr); sl = s.lower()
    state = ""
    for full, low in zip(INDIA_STATES, INDIA_STATES_L):
        if low in sl: state = full; break
    m = PIN_RE.search(s); pincode = m.group(1) if m else ""
    country = "India" if "india" in sl else ""
    return state, pincode, country

# -------------- main merge --------------
def merge_zip_to_json(zip_path: Path, relations_path: Path, out_path: Path):
    # Load relations (optional)
    rel_map = {}
    if relations_path and relations_path.exists():
        try:
            relations = json.loads(relations_path.read_text(encoding="utf-8"))
            for edge in relations:
                ftab = norm_table_name(edge.get("from_table","")); fcol = edge.get("from_column","")
                ttab = norm_table_name(edge.get("to_table",""));   tcol = edge.get("to_column","")
                if ftab and fcol and ttab and tcol:
                    rel_map[(ftab, fcol)] = (ttab, tcol)
        except Exception:
            pass

    # Load all CSVs
    with zipfile.ZipFile(zip_path, "r") as zf:
        tables = {}
        for member in zf.infolist():
            if member.is_dir(): continue
            if not member.filename.lower().endswith(".csv"): continue
            tname = norm_table_name(Path(member.filename).name)
            try:
                df = read_csv_from_zip(zf, member)
                tables[tname] = df
            except Exception:
                pass

    # Core tables
    company  = tables.get("CompanyMaster")
    products = tables.get("CompanyProducts")
    rdfac    = tables.get("CompanyRDFacility")
    tests    = tables.get("CompanyTestFacility")
    certs    = tables.get("CompanyCertificationDetail")
    turns    = tables.get("CompanyTurnOver")
    coreexp  = tables.get("CompanyCoreExpertiseMaster")

    if company is None or company.empty:
        raise SystemExit("CompanyMaster CSV not found or empty in ZIP")

    # Master lookups (added CompanyScale/OrganisationType masters)
    master_names = [
        "ProductTypeMaster","PlatformTechAreaMaster","DefencePlatformMaster",
        "RDCategoryMaster","RDSubCategoryMaster",
        "CertificationTypeMaster",
        "TestFacilityCategoryMaster","TestFacilitySubCategoryMaster",
        "YearMaster",
        "IndustryDomainMaster","IndustrySubdomainMaster","IndustrySubdomainType",
        "CoreExpertiseMaster",
        "CompanyScaleMaster","ScaleMaster",
        "OrganisationTypeMaster","OrganizationTypeMaster"
    ]
    lookups = { m: make_lookup(tables.get(m), id_col="Id") for m in master_names }

    def label_from(table_name: str, key: str) -> str:
        k = id_to_str(key)
        if not k: return ""
        return strip_spaces(lookups.get(table_name, {}).get(k, {}).get("_label",""))

    def label_from_any(table_names, key: str) -> str:
        for t in table_names:
            lab = label_from(t, key)
            if lab:
                return lab
        return ""

    # Index children
    prod_by_company = index_by_company(products)
    rd_by_company   = index_by_company(rdfac)
    test_by_company = index_by_company(tests)
    cert_by_company = index_by_company(certs)
    turn_by_company = index_by_company(turns)
    core_by_company = index_by_company(coreexp)

    out = {
        "metadata": {
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "source_zip": str(zip_path),
            "relations_file": str(relations_path),
            "tables_loaded": sorted(list(tables.keys())),
            "schema_version": "2.5",
            "description": "Integrated company JSON from ZIP CSVs with resolved foreign keys and cleaned values"
        },
        "companies": []
    }

    counts = {"product_records":0,"rd_facility_records":0,"test_facility_records":0,"certification_records":0,"turnover_records":0,"core_expertise_records":0}

    # --- coalescers for CompanyMaster ---
    def co_sc(c):  # company scale (raw/label)
        val = first_present(c, ["CompanyScale","CompanyScaleLabel","CompanyScale_Label","ScaleName","Scale"])
        if val: return val
        fk = first_present(c, ["CompanyScale_Fk_Id","CompanyScale_FK_ID","Scale_Fk_Id","ScaleId","ScaleID"])
        lab = label_from_any(["CompanyScaleMaster","ScaleMaster"], fk)
        return lab

    def co_ot(c):  # organization type (raw/label)
        val = first_present(c, ["OrganizationType","OrganizationTypeLabel","OrganizationType_Label","OrganisationType","OrganisationTypeLabel"])
        if val: return val
        fk = first_present(c, ["OrganizationType_Fk_Id","OrganizationType_FK_ID","OrganisationType_Fk_Id","OrganisationType_FK_ID","OrganisationTypeId","OrganisationTypeID"])
        lab = label_from_any(["OrganisationTypeMaster","OrganizationTypeMaster"], fk)
        return lab

    def co_ic(c, dom_labels):  # industrial classification
        val = first_present(c, ["IndustrialClassification","IndustryDomain","IndustryDomainText","Industry_Domain_Text"])
        if val: return val
        # fallback to first domain label if present
        if dom_labels: 
            return dom_labels[0]
        return ""

    def co_rd(c):  # registration_date
        raw = first_present(c, [
            "RegistrationDate","Date_Of_Registration","DateOfRegistration","DateOfIncorporation","IncorporationDate",
            "DateOfIncorp","RegDate","CompanyRegistrationDate"
        ])
        if not raw:
            return ""
        # try to normalize to dd-mm-YYYY (dayfirst)
        try:
            dt = pd.to_datetime(raw, dayfirst=True, errors="coerce")
            if pd.notna(dt):
                return dt.strftime("%d-%m-%Y")
        except Exception:
            pass
        return raw  # keep as-is if parsing fails

    for _, row in company.iterrows():
        c = {k: clean_str(row.get(k, "")) for k in company.columns}
        cid  = c.get("Id",""); cref = c.get("CompanyRefNo","")
        ckeys = [x for x in {cid, cref} if x]

        # Industry labels
        dom_labels, subdom_labels = [], []
        dom_fk = first_present(c, ["IndustryDomain_Fk_Id","IndustryDomain_FK_ID","IndustryDomainId","IndustryDomainID"])
        if dom_fk:
            lbl = label_from("IndustryDomainMaster", dom_fk)
            if lbl: dom_labels.append(lbl)
        sub_fk = first_present(c, ["IndustrySubdomain_Fk_Id","IndustrySubDomain_Fk_Id","IndustrySubdomainId","IndustrySubDomainID"])
        if sub_fk:
            lbl = label_from("IndustrySubdomainMaster", sub_fk)
            if lbl: subdom_labels.append(lbl)

        if not dom_labels:
            txt = first_present(c, ["IndustryDomain","industry_domain"])
            if txt: dom_labels = [t.strip() for t in re.split(r"[;,|]", txt) if t.strip()]
        if not subdom_labels:
            txt = first_present(c, ["IndustrySubDomain","industry_subdomain"])
            if txt: subdom_labels = [t.strip() for t in re.split(r"[;,|]", txt) if t.strip()]

        st_fallback, pin_fallback, country_fallback = infer_from_address(c.get("Address",""))

        obj = {
            "company_ref_no": cref,
            "company_id": cid,
            "company_name": c.get("CompanyName",""),
            "cin_number": c.get("CINNumber",""),
            "pan": c.get("Pan",""),
            "registration_date": co_rd(c),
            "company_status": c.get("CompanyStatus",""),
            "company_class": c.get("CompanyClass",""),
            "listing_status": c.get("ListingStatus",""),
            "company_category": c.get("CompanyCategory",""),
            "company_subcategory": c.get("CompanySubCategory",""),
            "industrial_classification": co_ic(c, dom_labels),
            "other_expertise": c.get("OtherExpertise",""),
            "other_industry_domain": c.get("OtherIndustryDomain",""),
            "other_industry_subdomain": c.get("OtherIndustrySubDomain",""),
            "address": c.get("Address",""),
            "city": c.get("CityName",""),
            "district": c.get("District",""),
            "state": c.get("StateName","") or st_fallback,
            "pincode": c.get("Pincode","") or pin_fallback,
            "email": c.get("EmailId",""),
            "poc_email": c.get("POC_Email",""),
            "phone": c.get("Phone",""),
            "website": c.get("WebsiteLink",""),
            "country": c.get("CountryName","") or country_fallback,
            "company_scale": strip_spaces(co_sc(c)),
            "organization_type": strip_spaces(co_ot(c)),
            "core_expertise_text": c.get("CoreExpertise",""),
            "industry_domain_text": c.get("IndustryDomain",""),
            "industry_subdomain_text": c.get("IndustrySubDomain",""),
            "industry": { "domains": dom_labels, "subdomains": subdom_labels },
            "products": [], "rd_facilities": [], "test_facilities": [], "certifications": [], "turnovers": [], "core_expertise": []
        }

        # Products
        rows = []
        for k in ckeys: rows += prod_by_company.get(k, [])
        for r in rows:
            r = {kk: clean_str(vv) for kk, vv in r.items()}
            p = {
                "product_ref_no": r.get("ProductRefNo",""),
                "product_name": r.get("ProductName",""),
                "product_description": first_present(r, ["ProductDesc","Description","Desc"]),
                "nsn_number": r.get("NSNNumber",""),
                "hsn_code": r.get("HSNCode",""),
                "annual_production_capacity": r.get("AnnualProductionCapacity",""),
                "salient_feature": first_present(r, ["SalientFeature","SalientFeatures"]),
                "product_type": strip_spaces(label_from("ProductTypeMaster", first_present(r, ["ProductType_Fk_Id","ProductType_FK_ID","ProductTypeId","ProductTypeID"]))),
                "platform_tech_area": strip_spaces(label_from("PlatformTechAreaMaster", first_present(r, ["PTAType_Fk_Id","PTAType_FK_ID","PTATypeId","PTATypeID"]))),
                "defence_platform": strip_spaces(label_from("DefencePlatformMaster", first_present(r, ["DefencePlatform_Fk_Id","DefencePlatform_FK_ID","DefencePlatformId","DefencePlatformID"])))
            }
            obj["products"].append(p)

        # R&D
        rows = []
        for k in ckeys: rows += rd_by_company.get(k, [])
        for r in rows:
            r = {kk: clean_str(vv) for kk, vv in r.items()}
            rd = {
                "rd_ref_no": r.get("RDRefNo",""),
                "rd_details": first_present(r, ["RD_Details","RDDetails","R_D_Details","Details","Description","Desc"]),
                "is_nabl_accredited": first_present(r, ["IsNablAccredited","IsNABLAccredited","IsNABL","NABLAccredited","Is_NABL"]),
                "rd_category": strip_spaces(label_from("RDCategoryMaster", first_present(r, ["RDCategory_Fk_Id","RDCategory_Fk_ID","RDCategoryId","RDCategoryID"]))),
                "rd_subcategory": strip_spaces(label_from("RDSubCategoryMaster", first_present(r, ["RDSubCategory_Fk_Id","RDSubCategory_Fk_ID","RDSubCategoryId","RDSubCategoryID"])))
            }
            obj["rd_facilities"].append(rd)

        # Tests
        rows = []
        for k in ckeys: rows += test_by_company.get(k, [])
        for r in rows:
            r = {kk: clean_str(vv) for kk, vv in r.items()}
            details = first_present(r, ["TF_Details","TestFacilityDetails","Details","Description","Desc"])
            test = {
                "test_ref_no": first_present(r, ["TFRefNo","TestRefNo","RefNo","TestFacilityRefNo"]),
                "test_details": details,
                "is_nabl_accredited": first_present(r, ["IsNablAccredited","IsNABLAccredited","IsNABL","NABLAccredited","Is_NABL"]),
                "category": strip_spaces(label_from("TestFacilityCategoryMaster", first_present(r, ["TestFacilityCategory_Fk_Id","TFCategory_Fk_Id","Category_Fk_Id","CategoryId","CategoryID"]))),
                "subcategory": strip_spaces(label_from("TestFacilitySubCategoryMaster", first_present(r, ["TestFacilitySubCategory_Fk_Id","TFSubCategory_Fk_Id","SubCategory_Fk_Id","SubCategoryId","SubCategoryID"]))),
            }
            obj["test_facilities"].append(test)

        # Certs
        rows = []
        for k in ckeys: rows += cert_by_company.get(k, [])
        for r in rows:
            r = {kk: clean_str(vv) for kk, vv in r.items()}
            cert = {
                "certificate_ref_no": first_present(r, ["CertRefNo","CertificateRefNo","RefNo"]),
                "certificate_no": first_present(r, ["CertificateNumber","CertificateNo","CertNo"]),
                "certificate_type": strip_spaces(label_from("CertificationTypeMaster", first_present(r, ["CertificationType_Fk_Id","CertificationType_FK_ID","CertificationTypeId","CertificationTypeID"]))),
                "issued_by": first_present(r, ["IssuedBy","Issuer","Authority"]),
                "valid_from": first_present(r, ["ValidFrom","IssueDate","StartDate"]),
                "valid_to": first_present(r, ["ValidTo","ExpiryDate","EndDate"]),
                "remarks": first_present(r, ["Remarks","Notes","Comment"])
            }
            obj["certifications"].append(cert)

        # Turnovers
        rows = []
        for k in ckeys: rows += turn_by_company.get(k, [])
        for r in rows:
            r = {kk: clean_str(vv) for kk, vv in r.items()}
            amt = first_present(r, ["TurnoverAmount","TurnOverAmount","Amount","Turnover","TurnOver"])
            year_fk = first_present(r, ["YearMaster_Fk_Id","Year_Fk_Id","YearId","YearID"])
            year_label = strip_spaces(label_from("YearMaster", year_fk))
            turnover = {"year": year_label or first_present(r, ["Year","YearText"]), "amount": strip_spaces(amt)}
            obj["turnovers"].append(turnover)

        # Core expertise
        rows = []
        for k in ckeys: rows += core_by_company.get(k, [])
        for r in rows:
            r = {kk: clean_str(vv) for kk, vv in r.items()}
            ce_fk = first_present(r, ["CoreExpertise_Fk_Id","CoreExpertise_FK_ID","CoreExpertiseId","CoreExpertiseID"])
            label = strip_spaces(label_from("CoreExpertiseMaster", ce_fk)) or first_present(r, ["CoreExpertiseName","CoreExpertise","Expertise","Name"])
            ce = {"core_expertise": strip_spaces(label or ""), "details": strip_spaces(first_present(r, ["Details","Description","Desc"]))}
            obj["core_expertise"].append(ce)

        # Final tidy top-level strings
        for key, val in list(obj.items()):
            if isinstance(val, str):
                obj[key] = strip_spaces(val)

        out["companies"].append(obj)

    # Counters
    for comp in out["companies"]:
        counts["product_records"] += len(comp["products"])
        counts["rd_facility_records"] += len(comp["rd_facilities"])
        counts["test_facility_records"] += len(comp["test_facilities"])
        counts["certification_records"] += len(comp["certifications"])
        counts["turnover_records"] += len(comp["turnovers"])
        counts["core_expertise_records"] += len(comp["core_expertise"])

    out["metadata"]["total_companies"] = len(out["companies"])
    out["metadata"].update(counts)

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to ZIP containing CSVs")
    ap.add_argument("--relations", required=True, help="Path to relations.json")
    ap.add_argument("--out", default="companies_merged.json", help="Output JSON path")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    rel_path = Path(args.relations)
    out_path = Path(args.out)
    if not zip_path.exists():
        raise SystemExit(f"ZIP not found: {zip_path}")
    if not rel_path.exists():
        raise SystemExit(f"relations.json not found: {rel_path}")
    res = merge_zip_to_json(zip_path, rel_path, out_path)
    print(f"Wrote {res}")

if __name__ == "__main__":
    main()
