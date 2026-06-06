import { useState } from "react"
import axios from "axios"

const TEAMS = [
  "Chennai Super Kings","Mumbai Indians",
  "Royal Challengers Bengaluru","Kolkata Knight Riders",
  "Sunrisers Hyderabad","Punjab Kings","Delhi Capitals",
  "Rajasthan Royals","Lucknow Super Giants","Gujarat Titans",
]
const TEAM_COLOR = {
  "Chennai Super Kings":         "#f59e0b",
  "Mumbai Indians":              "#3b82f6",
  "Royal Challengers Bengaluru": "#ef4444",
  "Kolkata Knight Riders":       "#8b5cf6",
  "Sunrisers Hyderabad":         "#f97316",
  "Punjab Kings":                "#dc2626",
  "Delhi Capitals":              "#2563eb",
  "Rajasthan Royals":            "#ec4899",
  "Lucknow Super Giants":        "#22c55e",
  "Gujarat Titans":              "#0ea5e9",
}
const QUICK = [
  ["Mumbai Indians","Chennai Super Kings"],
  ["Royal Challengers Bengaluru","Kolkata Knight Riders"],
  ["Sunrisers Hyderabad","Rajasthan Royals"],
  ["Delhi Capitals","Punjab Kings"],
  ["Lucknow Super Giants","Gujarat Titans"],
]

function WinBar({ data }) {
  const p1  = Math.round(data.team1_win_prob * 100)
  const p2  = 100 - p1
  const c1  = TEAM_COLOR[data.team1] || "#4f7bef"
  const c2  = TEAM_COLOR[data.team2] || "#ef4444"
  const conf= data.confidence || "low"

  return (
    <div style={{ marginTop:24 }}>
      {/* Team labels */}
      <div style={{ display:"flex", justifyContent:"space-between",
                    marginBottom:8, fontSize:13 }}>
        <span style={{ color:c1, fontWeight:600 }}>{data.team1}</span>
        <span style={{ color:"#666688", fontSize:11 }}>
          WIN PROBABILITY · {data.year}
        </span>
        <span style={{ color:c2, fontWeight:600 }}>{data.team2}</span>
      </div>

      {/* Probability bar */}
      <div style={{ display:"flex", height:48, borderRadius:10,
                    overflow:"hidden", border:"1px solid #252840" }}>
        <div style={{
          width:`${p1}%`, background:c1,
          display:"flex", alignItems:"center",
          justifyContent:"center",
          color:"#fff", fontWeight:800, fontSize:18,
          transition:"width .6s ease",
          flexShrink:0,
        }}>{p1}%</div>
        <div style={{
          flex:1, background:c2,
          display:"flex", alignItems:"center",
          justifyContent:"center",
          color:"#fff", fontWeight:800, fontSize:18,
        }}>{p2}%</div>
      </div>

      {/* Winner + confidence */}
      <div style={{ display:"flex", alignItems:"center",
                    justifyContent:"center", gap:12,
                    marginTop:14 }}>
        <span style={{ fontSize:16, fontWeight:700,
                       color:"#e8e8f0" }}>
          Predicted winner:&nbsp;
          <span style={{ color: p1>=50 ? c1 : c2 }}>
            {data.predicted_winner}
          </span>
        </span>
        <span className={`badge badge-${conf}`}>
          {conf} confidence
        </span>
      </div>

      {/* ELO info */}
      {data.projected_elos && (
        <div style={{ display:"flex", gap:12, marginTop:16 }}>
          {Object.entries(data.projected_elos).map(([team,elo]) => (
            <div key={team} style={{
              flex:1, background:"#111320",
              border:"1px solid #252840",
              borderRadius:8, padding:"10px 14px",
              textAlign:"center",
            }}>
              <div style={{ fontSize:11, color:"#9999bb",
                            marginBottom:4 }}>
                Projected ELO {data.year}
              </div>
              <div style={{ fontSize:20, fontWeight:700,
                            color: TEAM_COLOR[team]||"#fff" }}>
                {Math.round(elo)}
              </div>
              <div style={{ fontSize:11, color:"#666688",
                            marginTop:2 }}>{team}</div>
            </div>
          ))}
        </div>
      )}

      {/* Explanation */}
      {data.explanation?.length > 0 && (
        <div style={{ marginTop:16 }}>
          <div style={{ fontSize:11, color:"#666688",
                        textTransform:"uppercase",
                        letterSpacing:1, marginBottom:10 }}>
            Key factors
          </div>
          <div style={{ display:"flex", flexDirection:"column",
                        gap:8 }}>
            {data.explanation.map((e,i) => (
              <div key={i} style={{
                background:"#111320",
                border:"1px solid #252840",
                borderRadius:8, padding:"10px 14px",
                display:"flex", justifyContent:"space-between",
                alignItems:"center", gap:12,
              }}>
                <div>
                  <div style={{ fontSize:13, fontWeight:600,
                                color:"#e8e8f0" }}>{e.factor}</div>
                  <div style={{ fontSize:12, color:"#9999bb",
                                marginTop:2 }}>{e.detail}</div>
                </div>
                <div style={{
                  fontSize:11, fontWeight:600, whiteSpace:"nowrap",
                  color: e.favours===data.team1 ? c1 : c2,
                  background: "rgba(0,0,0,0.3)",
                  padding:"3px 10px", borderRadius:6,
                }}>
                  → {(e.favours||"").split(" ").slice(-1)[0]}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <p style={{ fontSize:11, color:"#444466",
                  textAlign:"center", marginTop:14 }}>
        {data.disclaimer}
      </p>
    </div>
  )
}

export default function MatchPredictor() {
  const [team1,   setTeam1]   = useState("Mumbai Indians")
  const [team2,   setTeam2]   = useState("Chennai Super Kings")
  const [year,    setYear]    = useState(2026)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function predict() {
    if (team1===team2) { setError("Select two different teams"); return }
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await axios.post("/api/predict/match",
        { team1, team2, year, venue:"neutral" })
      setResult(r.data)
    } catch(e) {
      setError(e.response?.data?.detail||"Prediction failed — is the backend running?")
    } finally { setLoading(false) }
  }

  return (
    <div>
      {/* Controls */}
      <div className="card">
        <div className="card-title">🎯 Select Match</div>

        {/* Team + year row */}
        <div style={{ display:"grid",
                      gridTemplateColumns:"1fr auto 1fr",
                      gap:12, alignItems:"end",
                      marginBottom:20 }}>
          <div>
            <label>Team 1</label>
            <select value={team1}
              onChange={e=>{setTeam1(e.target.value);setResult(null)}}>
              {TEAMS.map(t=><option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div style={{ paddingBottom:10, color:"#666688",
                        fontWeight:700, fontSize:14,
                        textAlign:"center" }}>VS</div>

          <div>
            <label>Team 2</label>
            <select value={team2}
              onChange={e=>{setTeam2(e.target.value);setResult(null)}}>
              {TEAMS.map(t=><option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        {/* Year slider */}
        <div style={{ marginBottom:20 }}>
          <label>Prediction Year — {year}</label>
          <div style={{ display:"flex", alignItems:"center",
                        gap:12 }}>
            <span style={{ fontSize:12, color:"#666688" }}>2025</span>
            <input type="range" min={2025} max={2049}
              value={year}
              onChange={e=>{setYear(Number(e.target.value));
                            setResult(null)}}
              style={{ flex:1, accentColor:"#4f7bef" }} />
            <span style={{ fontSize:12, color:"#666688" }}>2049</span>
          </div>
          <div style={{ textAlign:"center", marginTop:6,
                        fontSize:20, fontWeight:700,
                        color:"#4f7bef" }}>{year}</div>
        </div>

        <button className="btn-primary" onClick={predict}
          disabled={loading}>
          {loading
            ? <><div className="spinner"/> Predicting…</>
            : "🏏 Predict Match Winner"}
        </button>

        {error && <div className="alert-error">{error}</div>}
        {result && <WinBar data={result} />}
      </div>

      {/* Quick picks */}
      <div style={{ marginTop:20 }}>
        <div style={{ fontSize:11, color:"#666688",
                      textTransform:"uppercase",
                      letterSpacing:1, marginBottom:10 }}>
          Classic rivalries — quick pick
        </div>
        <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
          {QUICK.map(([t1,t2]) => (
            <button key={t1+t2} className="btn-sm"
              onClick={()=>{setTeam1(t1);setTeam2(t2);
                            setResult(null)}}>
              <span style={{ color:TEAM_COLOR[t1] }}>
                {t1.split(" ").pop()}
              </span>
              &nbsp;vs&nbsp;
              <span style={{ color:TEAM_COLOR[t2] }}>
                {t2.split(" ").pop()}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div className="card" style={{ marginTop:20 }}>
        <div className="card-title">ℹ️ How predictions work</div>
        <div style={{ display:"grid",
                      gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",
                      gap:12 }}>
          {[
            ["🔢 ELO Ratings","Team strength computed from 1076 matches. Updates after every result."],
            ["📊 Phase Features","Powerplay & death-over stats from ball-by-ball data. Strongest signal."],
            ["📅 Year Projection","Future ELO projected using team trajectory + cyclical form patterns."],
            ["⚖️ Calibration","Probabilities calibrated so 60% = wins 60% of the time."],
          ].map(([title,desc]) => (
            <div key={title} style={{
              background:"#111320", border:"1px solid #252840",
              borderRadius:8, padding:"12px 14px",
            }}>
              <div style={{ fontSize:13, fontWeight:600,
                            color:"#e8e8f0",marginBottom:4 }}>
                {title}
              </div>
              <div style={{ fontSize:12, color:"#9999bb",
                            lineHeight:1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
