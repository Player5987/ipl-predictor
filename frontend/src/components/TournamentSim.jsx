// ── TournamentSim.jsx ────────────────────────────────────
import { useState } from "react"
import axios from "axios"

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

function Medal({ rank }) {
  if (rank===1) return <span title="Champion">🥇</span>
  if (rank===2) return <span title="Runner-up">🥈</span>
  if (rank===3) return <span>🥉</span>
  return <span style={{ color:"#444466", fontWeight:700,
                        fontSize:14 }}>#{rank}</span>
}

export function TournamentSim() {
  const [year,    setYear]    = useState(2026)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [view,    setView]    = useState("standings")

  async function simulate() {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await axios.get(`/api/predict/tournament/${year}`)
      setResult(r.data)
    } catch(e) {
      setError(e.response?.data?.detail||"Simulation failed")
    } finally { setLoading(false) }
  }

  // Build matrix lookup
  const mm = {}
  if (result) for (const m of result.matchups) {
    mm[`${m.team1}__${m.team2}`] = m.team1_prob
    mm[`${m.team2}__${m.team1}`] = m.team2_prob
  }
  const teams = result?.predicted_standings?.map(s=>s.team)||[]
  const maxW   = result
    ? Math.max(...result.predicted_standings.map(s=>s.expected_wins))
    : 9

  return (
    <div>
      {/* Control */}
      <div className="card">
        <div className="card-title">⚙️ Simulation Settings</div>
        <div style={{ display:"flex", gap:16,
                      alignItems:"flex-end", flexWrap:"wrap" }}>
          <div style={{ flex:1, minWidth:240 }}>
            <label>Season Year — {year}</label>
            <input type="range" min={2025} max={2049}
              value={year}
              onChange={e=>setYear(Number(e.target.value))}
              style={{ width:"100%", accentColor:"#4f7bef",
                       marginTop:6 }} />
            <div style={{ display:"flex",
                          justifyContent:"space-between",
                          fontSize:11, color:"#666688" }}>
              <span>2025</span><span>2049</span>
            </div>
          </div>
          <button className="btn-primary"
            onClick={simulate} disabled={loading}
            style={{ width:"auto", padding:"12px 32px" }}>
            {loading
              ? <><div className="spinner"/>Simulating…</>
              : `🏆 Simulate ${year} Season`}
          </button>
        </div>
        {error && <div className="alert-error">{error}</div>}
      </div>

      {result && (
        <>
          {/* View toggle */}
          <div style={{ display:"flex", gap:8, margin:"16px 0" }}>
            {[["standings","🏆 Standings"],
              ["matrix","🔢 Head-to-Head Matrix"]
            ].map(([v,l])=>(
              <button key={v}
                className={`btn-sm ${view===v?"btn-sm--active":""}`}
                onClick={()=>setView(v)}>{l}</button>
            ))}
          </div>

          {/* STANDINGS */}
          {view==="standings" && (
            <div className="card">
              <div className="card-title">
                📊 Predicted Standings — IPL {result.year}
              </div>
              <div style={{ display:"flex",
                            flexDirection:"column", gap:8 }}>
                {result.predicted_standings.map((s,i)=>{
                  const color = TEAM_COLOR[s.team]||"#4f7bef"
                  const pct   = (s.expected_wins/maxW)*100
                  const elo_delta = s.elo_vs_2024
                  return (
                    <div key={s.team} style={{
                      display:"flex", alignItems:"center",
                      gap:14, padding:"14px 16px",
                      background:"#111320",
                      border:"1px solid #252840",
                      borderRadius:10,
                    }}>
                      {/* Rank */}
                      <div style={{ width:36, textAlign:"center",
                                    fontSize:20 }}>
                        <Medal rank={i+1} />
                      </div>
                      {/* Color bar */}
                      <div style={{ width:4, height:44,
                                    background:color,
                                    borderRadius:2, flexShrink:0 }}/>
                      {/* Name */}
                      <div style={{ flex:1 }}>
                        <div style={{ fontSize:14, fontWeight:600,
                                      color:"#e8e8f0" }}>
                          {s.team}
                        </div>
                        <div style={{ fontSize:11,
                                      color:"#666688",
                                      marginTop:2 }}>
                          ELO {Math.round(s.elo)}&nbsp;
                          {elo_delta!==undefined && (
                            <span style={{
                              color: elo_delta>=0
                                ?"#22c55e":"#ef4444"
                            }}>
                              ({elo_delta>=0?"+":""}{elo_delta})
                            </span>
                          )}
                          &nbsp;·&nbsp;
                          {s.trajectory>0?"↑ improving":
                           s.trajectory<0?"↓ declining":
                           "→ stable"}
                        </div>
                      </div>
                      {/* Expected wins */}
                      <div style={{ textAlign:"right",
                                    minWidth:64 }}>
                        <div style={{ fontSize:22, fontWeight:700,
                                      color }}>
                          {s.expected_wins.toFixed(1)}
                        </div>
                        <div style={{ fontSize:11,
                                      color:"#666688" }}>
                          exp. wins
                        </div>
                      </div>
                      {/* Bar */}
                      <div style={{ width:120 }}>
                        <div className="prog-wrap">
                          <div className="prog-bar"
                            style={{ width:`${pct}%`,
                                     background:color }}/>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
              <p style={{ fontSize:11, color:"#444466",
                           marginTop:14, textAlign:"center" }}>
                {result.n_matchups} matchups simulated ·
                Expected wins out of 9 league games
              </p>
            </div>
          )}

          {/* MATRIX */}
          {view==="matrix" && (
            <div className="card" style={{ overflowX:"auto" }}>
              <div className="card-title">
                🔢 Head-to-Head Win Probability Matrix
              </div>
              <p style={{ fontSize:12, color:"#9999bb",
                           marginBottom:12 }}>
                Cell = probability that the <strong>row</strong>
                &nbsp;team beats the <strong>column</strong> team
              </p>
              <table style={{ borderCollapse:"collapse",
                              fontSize:11, minWidth:560 }}>
                <thead>
                  <tr>
                    <th style={{ padding:"8px 10px",
                                  color:"#666688",
                                  textAlign:"left",
                                  borderBottom:"1px solid #252840",
                                  minWidth:120 }}>Team</th>
                    {teams.map(t=>(
                      <th key={t} style={{
                        padding:"4px 6px", color:"#666688",
                        fontSize:10, textAlign:"center",
                        borderBottom:"1px solid #252840",
                        minWidth:52,
                      }}>
                        <div style={{
                          width:6, height:6, borderRadius:"50%",
                          background:TEAM_COLOR[t]||"#4f7bef",
                          margin:"0 auto 3px",
                        }}/>
                        {t.split(" ").pop()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {teams.map(t1=>(
                    <tr key={t1}>
                      <td style={{ padding:"8px 10px",
                                    fontWeight:600,
                                    color:"#9999bb",
                                    fontSize:11,
                                    whiteSpace:"nowrap",
                                    borderBottom:"1px solid #1a1d2e" }}>
                        <span style={{
                          display:"inline-block", width:6, height:6,
                          borderRadius:"50%",
                          background:TEAM_COLOR[t1]||"#4f7bef",
                          marginRight:6,
                        }}/>
                        {t1.split(" ").pop()}
                      </td>
                      {teams.map(t2=>{
                        if (t1===t2) return (
                          <td key={t2} style={{
                            textAlign:"center", color:"#333350",
                            background:"#111320",
                            borderBottom:"1px solid #1a1d2e",
                            padding:"8px 6px",
                          }}>—</td>
                        )
                        const p = mm[`${t1}__${t2}`]
                        const pct = p ? Math.round(p*100) : 50
                        const bg  = pct>=60?"rgba(34,197,94,.12)":
                                    pct>=50?"rgba(79,123,239,.08)":
                                            "rgba(239,68,68,.08)"
                        const col = pct>=60?"#4ade80":
                                    pct>=50?"#93c5fd":"#fca5a5"
                        return (
                          <td key={t2} style={{
                            textAlign:"center",
                            background:bg, color:col,
                            fontWeight:600, fontSize:12,
                            padding:"8px 6px",
                            borderBottom:"1px solid #1a1d2e",
                          }}>
                            {pct}%
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ display:"flex", gap:20,
                            marginTop:12, fontSize:11 }}>
                <span>
                  <span style={{ color:"#4ade80" }}>■</span>
                  &nbsp;60%+ (strong favourite)
                </span>
                <span>
                  <span style={{ color:"#93c5fd" }}>■</span>
                  &nbsp;50-60% (slight edge)
                </span>
                <span>
                  <span style={{ color:"#fca5a5" }}>■</span>
                  &nbsp;Under 50% (underdog)
                </span>
              </div>
            </div>
          )}
        </>
      )}

      {/* Info box */}
      <div className="card" style={{ marginTop:16 }}>
        <div className="card-title">ℹ️ How the simulator works</div>
        <div className="grid-3">
          {[
            ["45 Matchups","Simulates all possible team pairings in round-robin format"],
            ["ELO Projection","Each team's rating projected to the chosen year using trajectory models"],
            ["Expected Wins","Sum of win probabilities across all matchups gives predicted league rank"],
          ].map(([t,d])=>(
            <div key={t} style={{
              background:"#111320", border:"1px solid #252840",
              borderRadius:8, padding:"12px 14px",
            }}>
              <div style={{ fontWeight:600, color:"#e8e8f0",
                            fontSize:13, marginBottom:4 }}>{t}</div>
              <div style={{ fontSize:12, color:"#9999bb",
                            lineHeight:1.5 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default TournamentSim
