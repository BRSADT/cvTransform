import os, re, json, glob, argparse
from dataclasses import dataclass
from typing import Dict, Optional

from PyPDF2 import PdfReader

# --------- Utilidades PDF ---------
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        parts.append(t)
    txt = "\n".join(parts).replace("\r", "\n")
    # No "clean" agresivo: sólo normalizamos saltos del SO
    return txt

# --------- Esquema ---------
SCHEMA_KEYS = ["header", "resumen", "experiencia", "educacion", "skills"]

@dataclass
class CvStruct:
    header: str
    resumen: str
    experiencia: str
    educacion: str
    skills: str

def ensure_schema(d: Dict) -> CvStruct:
    # NO limpiar: mantenemos texto tal cual; sólo convertimos a str si viene otro tipo.
    def g(k):
        v = d.get(k, "")
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        # Normaliza saltos de línea del SO, sin strip ni cambios de espacios/puntuación
        return v.replace("\r", "\n")
    return CvStruct(
        header=g("header"),
        resumen=g("resumen"),
        experiencia=g("experiencia"),
        educacion=g("educacion"),
        skills=g("skills"),
    )

# --------- Render TXT (ATS) ----------
def render_txt(cv: CvStruct) -> str:
    parts = []
    if cv.header.strip():
        parts.append(cv.header.strip())

    parts.append("\nResumen\n-------")
    parts.append(cv.resumen)

    parts.append("\nExperiencia\n-----------")
    parts.append(cv.experiencia)

    parts.append("\nEducación\n---------")
    parts.append(cv.educacion)

    # Skills: encabezado SOLO una vez. Si el contenido ya trae "Skills\n------"
    # (p.ej., porque el modelo lo copió textual), no lo repetimos.
    skills_body = cv.skills or ""
    if skills_body.startswith("Skills\n------"):
        parts.append("")  # separador
        parts.append(skills_body.rstrip())
    else:
        parts.append("\nSkills\n------")
        parts.append(skills_body)

    # No hacemos 'cleaning' adicional.
    out = "\n".join(parts).rstrip() + "\n"
    return out

# --------- Prompt ----------
# Nota: aquí NO pedimos que el modelo agregue "Skills\\n------" en el valor de 'skills'.
# Pedimos que 'skills' contenga SOLO el contenido de la sección explícita (sin encabezado),
# para que este encabezado lo añada el renderer y evitar duplicados.
SYS_PROMPT = (
"Devuelve SOLO un JSON con claves: header, resumen, experiencia, educacion, skills. "
"Cada valor debe ser STRING y usar \\n para saltos. NO INVENTES: solo texto que aparezca literalmente en el CV.\n"
"- (Inferencia de secciones): detecta cada sección aunque el CV use sinónimos/títulos distintos: "
"  Resumen↔Perfil/About/Summary, Experiencia↔Experience/Trayectoria/Laboral, "
"  Educación↔Formación/Education/Estudios, Skills↔Habilidades/Competencias.\n"
"- header: 'Nombre — Rol' si el rol aparece; luego solo si existen (una por línea): Email:, LinkedIn:, Web:, Tel:, Ubicación:.\n"
"- resumen: copia si existe; si no, \"\".\n"
"- experiencia: texto plano; conserva viñetas ('- ' o '•').\n"
"- educacion: una entrada por línea; si hay grado/institución/año juntos o contiguos, usa 'Grado — Institución (Año)'; sin comas finales.\n"
"- skills: INCLUYE solo si existe una sección explícita de Skills/Habilidades/Competencias (no extraigas desde Experiencia ni Educación). "
"  Si hay habilidades en el CV, devuelve EXACTAMENTE: "
"  'Skills\\n------\\n' + ('Hard skills: a, b, c\\n' si hay hard) + ('Soft skills: x, y, z' si hay soft). "
"  Las listas SOLO con términos que aparezcan en el CV, sin duplicados (case-insensitive), separados por coma+espacio, sin coma/punto final. "
"  Si NO hay ninguna habilidad encontrada o no existe la sección explícita, pon skills=\"\". NUNCA repitas 'Skills\\n------'.\n"
"Normaliza comas/espacios: 1 espacio tras cada coma; ninguna coma antes de ')'. "
"Salida: SOLO el objeto JSON válido, sin Markdown ni comentarios."
)




USER_WRAP = lambda body: f"---CV---\n{body}\n---FIN---"

# --------- Parser robusto JSON ----------
def extract_json_text(s: str) -> str:
    s = s.strip()
    # quita fences tipo ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()

    # encuentra primer objeto { ... } balanceado
    start = s.find("{")
    if start == -1:
        return s  # no hay { => dejará fallar en json.loads para depurar
    stack = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            else:
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    return s[start:i+1]
    return s[start:]  # si no cerró, intentamos igual

def _escape_newlines_in_strings(text: str) -> str:
    out, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                out.append(ch); esc = False
            else:
                if ch == "\\":
                    out.append(ch); esc = True
                elif ch == '"':
                    out.append(ch); in_str = False
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    continue
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)

def parse_llm_json(s: str) -> Dict:
    s = extract_json_text(s)
    # comillas tipográficas → ascii
    s = (s.replace("“","\"").replace("”","\"").replace("„","\"")
           .replace("’","'").replace("‘","'"))
    # comas colgantes tipo ,} o ,]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # asegura \n escapados dentro de strings
    s = _escape_newlines_in_strings(s)
    return json.loads(s)

# --------- Proveedores (Gemini / OpenAI) ---------
def call_gemini(text: str, model: str = "models/gemini-1.5-flash") -> str:
    import google.generativeai as genai
    api = os.getenv("GEMINI_API_KEY")
    if not api:
        raise RuntimeError("Falta GEMINI_API_KEY")
    genai.configure(api_key=api)

    gmodel = genai.GenerativeModel(
        model,
        system_instruction=SYS_PROMPT,
        generation_config={"response_mime_type": "application/json", "temperature": 0}
    )
    # Un solo mensaje de usuario con el CV envuelto
    resp = gmodel.generate_content(USER_WRAP(text))
    return resp.text

def call_openai(text: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("Falta OPENAI_API_KEY")
    client = OpenAI(api_key=api)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},  # fuerza JSON
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

    # Parseo/normalización mínima
    data = parse_llm_json(out)
    cv = ensure_schema(data)

    # Render TXT
    base = f"{idx:04d}"
    if suffix:
        base = f"{base}_{suffix}"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_txt(cv))
    return out_path

def main():
    ap = argparse.ArgumentParser(description="PDF → TXT (formato ATS) usando API LLM (Gemini/OpenAI).")
    ap.add_argument("--in_glob", required=True, help="Patrón de PDFs, p.ej. './CV/*.pdf'")
    ap.add_argument("--outdir", default="./salida")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--suffix", default="", help="ej: exito, fracaso_med (se agrega al nombre)")
    ap.add_argument("--provider", choices=["gemini","openai"], default="gemini")
    ap.add_argument("--model", default=None, help="Override de modelo (opcional)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    if not files:
        print("No se encontraron PDFs.")
        return

    idx = args.start
    for p in files:
        try:
            outp = process_one(p, args.outdir, idx, args.provider, args.suffix, args.model)
            print(f"[OK] {p} -> {outp}")
        except Exception as e:
            print(f"[ERR] {p}: {e}")
        idx += 1

if __name__ == "__main__":
    main()