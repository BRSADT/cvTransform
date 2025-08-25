# -*- coding: utf-8 -*-
import os, re, json, glob, argparse
from dataclasses import dataclass
from typing import Dict, Optional

from PyPDF2 import PdfReader
#import google.generativeai as genai
# --------- Utilidades PDF ---------
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        parts.append(t)
    txt = "\n".join(parts).replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# --------- Esquema y normalización ---------
SCHEMA_KEYS = [
    "header", "resumen", "experiencia", "educacion", "skills"
]

@dataclass
class CvStruct:
    header: str
    resumen: str
    experiencia: str
    educacion: str
    skills: str

def ensure_schema(d: Dict) -> CvStruct:
    # Fills missing fields with empty string; cleans extra whitespace
    def g(k):
        v = d.get(k, "")
        v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        v = v.replace("\r", "\n")
        v = re.sub(r"[^\S\n]+", " ", v)
        v = re.sub(r"\n{3,}", "\n\n", v)
        return v.strip()
    return CvStruct(
        header=g("header"),
        resumen=g("resumen"),
        experiencia=g("experiencia"),
        educacion=g("educacion"),
        skills=g("skills"),
    )

def render_txt(cv: CvStruct) -> str:
    def nz(x): return x if x.strip() else "(vacío)"
    parts = []
    if cv.header.strip():
        parts.append(cv.header.strip())
    parts.append("\nResumen\n-------")
    parts.append(nz(cv.resumen))
    parts.append("\nExperiencia\n-----------")
    parts.append(nz(cv.experiencia))
    parts.append("\nEducación\n---------")
    parts.append(nz(cv.educacion))
    parts.append("\nSkills\n------")
    parts.append(nz(cv.skills))
    return ("\n".join(parts)).strip() + "\n"

# --------- Prompts ----------
SYS_PROMPT = (
"Devuelve SOLO un JSON con claves: header, resumen, experiencia, educacion, skills. "
"Cada valor debe ser STRING y usar \\n para saltos. NO INVENTES NI INFI ERAS: solo texto literal del CV.\n"
"- header: 'Nombre — Rol' si el rol aparece explícitamente; luego solo si existen (una por línea): Email:, LinkedIn:, Web:, Tel:, Ubicación:.\n"
"- resumen: copia textual si existe; si no, \"\".\n"
"- experiencia: copia textual; conserva bullets/viñetas tal cual; no agregues ni quites puntuación.\n"
"- educacion: copia textual; no reformatees (no cambies comas, guiones, paréntesis, ni orden).\n"
"- skills: SOLO si el CV tiene una sección explícita de Skills/Habilidades/Competencias. "
"  Si NO existe esa sección, skills=\"\". "
"  Si existe, imprime UNA sola vez el encabezado exacto: 'Skills\\n------\\n'. "
"  Luego:\n"
"    • Si el CV trae líneas 'Hard skills:' y/o 'Soft skills:', CÓPIALAS textual (con duplicados, faltas de ortografía y comas tal como están).\n"
"    • Si NO trae esas líneas, copia el contenido de la sección de skills TAL CUAL (sin crear ni clasificar hard/soft).\n"
"- NUNCA extraigas skills desde 'Experiencia' o 'Educación'.\n"
"Salida: SOLO el objeto JSON válido (sin Markdown, sin comentarios, sin código)."
)

USER_WRAP = lambda body: f"---CV---\n{body}\n---FIN---"

def extract_json_text(s: str) -> str:
    # Quita posibles fences ```json ... ```
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Intenta encontrar el primer {...} balanceado
    m = re.search(r"\{.*\}\s*$", s, flags=re.S)
    return m.group(0) if m else s

def parse_llm_json(s: str) -> Dict:
    s = extract_json_text(s)
    # Arreglos comunes: comillas raras, trailing commas
    s = s.replace("“","\"").replace("”","\"").replace("’","'")
    # Quita comas colgantes
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return json.loads(s)

# --------- Proveedores (Gemini / OpenAI) ---------
def call_gemini(text: str, model: str = "models/gemini-1.5-flash") -> str:
    import google.generativeai as genai
    api = os.getenv("GEMINI_API_KEY")
    if not api:
        raise RuntimeError("Falta GEMINI_API_KEY")
    genai.configure(api_key=api)
    gmodel = genai.GenerativeModel(model)
    resp = gmodel.generate_content([SYS_PROMPT, USER_WRAP(text)])
    return resp.text

def call_openai(text: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("Falta OPENAI_API_KEY")
    client = OpenAI(api_key=api)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":SYS_PROMPT},
            {"role":"user","content":USER_WRAP(text)}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

# --------- Orquestación CLI ---------
def process_one(pdf_path: str, outdir: str, idx: int, provider: str, suffix: str, model_name: Optional[str]) -> str:
    raw = read_pdf(pdf_path)
    # Llama proveedor
    if provider == "gemini":
        out = call_gemini(raw, model=model_name or "models/gemini-1.5-flash")
    elif provider == "openai":
        out = call_openai(raw, model=model_name or "gpt-4o-mini")
    else:
        raise ValueError("provider debe ser 'gemini' u 'openai'")

    # Parseo/normalización
    data = parse_llm_json(out)
    cv = ensure_schema(data)

    # Render TXT
    base = f"{idx:04d}"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_txt(cv))
    return out_path

def main():
    ap = argparse.ArgumentParser(description="PDF → TXT (formato Claudia) usando API LLM.")
    ap.add_argument("--in_glob", required=True, help="Patrón de PDFs, p.ej. './CV/*.pdf'")
    ap.add_argument("--outdir", default="./salida")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--suffix", default="", help="ej: exito, fracaso_med (solo para el nombre)")
    ap.add_argument("--provider", choices=["gemini","openai"], default="gemini")
    ap.add_argument("--model", default=None, help="Override de modelo (opcional)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    if not files:
        print("No se encontraron PDFs.")
        return

    idx = args.start
    for p in files:
        outp = process_one(p, args.outdir, idx, args.provider, args.suffix, args.model)
        print(f"[OK] {p} -> {outp}")
        idx += 1

if __name__ == "__main__":
    main()
