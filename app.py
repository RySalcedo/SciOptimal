import streamlit as st
from supabase import create_client
from datetime import datetime
import json
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Plateau Breaker Engine",
    page_icon="🏋️",
    layout="centered"
)

# -----------------------------
# SECRETS / CLIENTS
# -----------------------------
SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_key"]
GEMINI_API_KEY = st.secrets["gemini_api_key"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")


# -----------------------------
# DATA ACCESS (SUPABASE)
# -----------------------------
def save_lift_history(email: str, lift_name: str, result: dict):
    """Insert a new history row into Supabase."""
    supabase.table("history").insert({
        "email": email,
        "lift_name": lift_name,
        "result": result,
        "created_at": datetime.utcnow().isoformat()
    }).execute()


def load_lift_history(email: str):
    """Load all history rows for a given user."""
    response = (
        supabase.table("history")
        .select("*")
        .eq("email", email)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


# -----------------------------
# AI LOGIC
# -----------------------------
SYSTEM_PROMPT = """
You are a strength coach helping a lifter break plateaus.
Given a lift, recent performance, and context, you will:

1. Analyze why they might be stuck.
2. Suggest specific changes (volume, intensity, frequency, exercise selection).
3. Provide a 1–2 week microcycle focused on breaking the plateau.
4. Keep language clear, direct, and practical.

Return your answer as structured JSON with keys:
- "summary": short text
- "diagnosis": list of bullet points
- "recommendations": list of bullet points
- "microcycle": list of days, each with "day", "focus", "exercises"
"""


def analyze_lift(lift_name: str, weight: float, reps: int, rpe: float, notes: str):
    user_prompt = f"""
Lift: {lift_name}
Top set: {weight} lbs x {reps} reps @ RPE {rpe}
Notes: {notes}

Using the instructions, respond ONLY with valid JSON.
"""
    response = model.generate_content(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}]
    )

    # Try to parse JSON; if it fails, wrap raw text
    try:
        text = response.text.strip()
        result = json.loads(text)
    except Exception:
        result = {
            "summary": "AI response could not be parsed as JSON.",
            "diagnosis": [response.text],
            "recommendations": [],
            "microcycle": []
        }
    return result


# -----------------------------
# UI HELPERS
# -----------------------------
def render_result(result: dict):
    st.subheader("Summary")
    st.write(result.get("summary", ""))

    diagnosis = result.get("diagnosis", [])
    if diagnosis:
        st.subheader("Diagnosis")
        for item in diagnosis:
            st.markdown(f"- {item}")

    recs = result.get("recommendations", [])
    if recs:
        st.subheader("Recommendations")
        for item in recs:
            st.markdown(f"- {item}")

    micro = result.get("microcycle", [])
    if micro:
        st.subheader("Suggested Microcycle")
        for day in micro:
            day_name = day.get("day", "Day")
            focus = day.get("focus", "")
            exercises = day.get("exercises", [])
            st.markdown(f"**{day_name}** — {focus}")
            for ex in exercises:
                st.markdown(f"- {ex}")
            st.markdown("---")


def render_history(history_rows):
    if not history_rows:
        st.info("No history yet. Run an analysis to create your first entry.")
        return

    st.subheader("History")
    for row in history_rows:
        created = row.get("created_at", "")
        lift_name = row.get("lift_name", "")
        result = row.get("result", {})
        with st.expander(f"{lift_name} — {created}"):
            render_result(result)


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("🏋️ Plateau Breaker Engine")
    st.caption("Supabase + Gemini + Streamlit")

    # Simple "auth": just email field for now
    email = st.text_input("Email (used to save your history)")
    if not email:
        st.warning("Enter your email to use history.")
        st.stop()

    st.markdown("---")
    st.header("New Analysis")

    col1, col2 = st.columns(2)
    with col1:
        lift_name = st.text_input("Lift name", value="Bench Press")
        weight = st.number_input("Top set weight (lbs)", min_value=0.0, value=225.0, step=5.0)
    with col2:
        reps = st.number_input("Reps", min_value=1, value=5, step=1)
        rpe = st.number_input("RPE", min_value=5.0, max_value=10.0, value=8.5, step=0.5)

    notes = st.text_area("Context / notes (optional)", placeholder="Sleep, fatigue, recent changes, etc.")

    if st.button("Analyze plateau", type="primary"):
        with st.spinner("Analyzing with Gemini..."):
            result = analyze_lift(lift_name, weight, reps, rpe, notes)
            save_lift_history(email, lift_name, result)
        st.success("Analysis complete and saved to history.")
        render_result(result)

    st.markdown("---")
    st.header("Your History")
    history_rows = load_lift_history(email)
    render_history(history_rows)


if __name__ == "__main__":
    main()
