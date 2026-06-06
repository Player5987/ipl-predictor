import { useState, useEffect } from "react"
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

const TEAM_INFO = {
  "Chennai Super Kings":         { titles:5, founded:2008, home:"Chennai" },
  "Mumbai Indians":              { titles:5, founded:2008, home:"Mumbai"  },
  "Royal Challengers Bengaluru": { titles:1, founded:2008, home:"Bengaluru" },
  "Kolkata Knight Riders":       { titles:3, founded:2008, home:"Kolkata" },
  "Sunrisers Hyderabad":         { titles:2, founded:2012, home:"Hyderabad" },
  "Punjab Kings":                { titles:0, founded:2008, home:"Chandigarh" },
  "Delhi Capitals":              { titles:0, founded:2008, home:"Delhi"   },
  "Rajasthan Royals":            { titles:2, founded:2008, home:"Jaipur"  },
  "Lucknow Super Giants":        { titles:0, founded:2022, home:"Lucknow" },
  "Gujarat Titans":              { titles:1, founded:2022, home:"Ahmedabad" },
}

export default function EloRankings() {
  const [rankings, setRankings] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)
  const [selected, setSelected] = useState(null)

  useEffect(()=>{
    axios.get("/api/elo/rankings")
      .then(r=>{ setRankings(r.data.rankings||[]); setLoading(false) })
      .catch(()=>{ setError("Could not load — is backend running?")
                   setLoading(false) })
  },[])

  const maxElo = rankings.length
    ? Math.max(...rankings.map(r=>r.elo)) : 1700
  const minElo = rankings.length
    ? Math.min(...rankings.map(r=>r.elo)) : 1400

  if (loading) return (
    <div className="loading">
      <div className="spinner"/>Loading ELO rankings…
    </div>
  )
  if (error) return <div className="alert-error">{error}</div>

  return (
    <div>
      {/* What is ELO */}
      <div className="card">
        <div className="card-title">📖 About ELO Ratings</div>
        <div className="grid-3">
          {[
            ["How it works","ELO updates after every match. Beating a strong team earns more points than beating a weak team."],
            ["Starting value","All teams start at 1500. After 18 seasons the spread reflects real historical performance."],
            ["Interpretation","ELO 1600 = wins ~58% of matches vs average opposition. ELO 1400 = wins ~42%."],
          ].map(([t,d])=>(
            <div key={t} style={{
              background:"#111320", border:"1px solid #252840",
              borderRadius:8, padding:"12px 14px",
            }}>
              <div style={{ fontWeight:600, fontSize:13,
                            color:"#e8e8f0", marginBottom:4 }}>{t}</div>
              <div style={{ fontSize:12, color:"#9999bb",
                            lineHeight:1.5 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Rankings */}
      <div className="card" style={{ marginTop:16 }}>
        <div className="card-title">⚡ IPL Power Rankings</div>
        <div style={{ display:"flex", flexDirection:"column",
                      gap:10 }}>
          {rankings.map((r,i)=>{
            const color   = TEAM_COLOR[r.team]||"#4f7bef"
            const info    = TEAM_INFO[r.team]||{}
            const pct     = ((r.elo-minElo)/(maxElo-minElo+1))*100
            const winPct  = Math.round(
              1/(1+Math.pow(10,(1500-r.elo)/400))*100)
            const isOpen  = selected===r.team

            return (
              <div key={r.team}>
                <div
                  onClick={()=>setSelected(
                    isOpen ? null : r.team)}
                  style={{
                    display:"flex", alignItems:"center",
                    gap:14, padding:"16px",
                    background: isOpen?"#1a1d2e":"#111320",
                    border:`1px solid ${isOpen
                      ? color+"44":"#252840"}`,
                    borderRadius: isOpen?"10px 10px 0 0":10,
                    cursor:"pointer", transition:"all .15s",
                  }}>
                  {/* Rank number */}
                  <div style={{
                    width:40, height:40, borderRadius:8,
                    background: i===0?"rgba(245,158,11,.15)":
                                i===1?"rgba(160,160,160,.15)":
                                i===2?"rgba(205,127,50,.15)":
                                       "#0b0d14",
                    border:`1px solid ${
                      i===0?"rgba(245,158,11,.3)":
                      i===1?"rgba(160,160,160,.3)":
                      i===2?"rgba(205,127,50,.3)":"#252840"}`,
                    display:"flex", alignItems:"center",
                    justifyContent:"center",
                    fontSize: i<3?18:14,
                    fontWeight:700,
                    color: i===0?"#fbbf24":
                           i===1?"#d1d5db":
                           i===2?"#d97706":"#444466",
                    flexShrink:0,
                  }}>
                    {i===0?"🥇":i===1?"🥈":i===2?"🥉":i+1}
                  </div>

                  {/* Color stripe */}
                  <div style={{
                    width:4, height:44, background:color,
                    borderRadius:2, flexShrink:0,
                  }}/>

                  {/* Team name */}
                  <div style={{ flex:1 }}>
                    <div style={{ fontSize:15, fontWeight:600,
                                  color:"#e8e8f0" }}>{r.team}</div>
                    <div style={{ fontSize:11, color:"#666688",
                                  marginTop:2 }}>
                      Wins {winPct}% vs avg team
                      {info.titles>0 &&
                        ` · ${info.titles}× IPL Champion`}
                    </div>
                  </div>

                  {/* ELO score */}
                  <div style={{ textAlign:"right",
                                minWidth:70 }}>
                    <div style={{ fontSize:26, fontWeight:800,
                                  color, lineHeight:1 }}>
                      {Math.round(r.elo)}
                    </div>
                    <div style={{ fontSize:11,
                                  color:"#666688" }}>ELO</div>
                  </div>

                  {/* Progress bar */}
                  <div style={{ width:130 }}>
                    <div className="prog-wrap">
                      <div className="prog-bar"
                        style={{ width:`${pct}%`,
                                 background:color }}/>
                    </div>
                  </div>

                  {/* Chevron */}
                  <div style={{
                    color:"#444466", fontSize:14,
                    transform: isOpen?"rotate(180deg)":"none",
                    transition:"transform .2s",
                  }}>▼</div>
                </div>

                {/* Expanded info */}
                {isOpen && (
                  <div style={{
                    background:"#1a1d2e",
                    border:`1px solid ${color}44`,
                    borderTop:"none",
                    borderRadius:"0 0 10px 10px",
                    padding:"16px",
                  }}>
                    <div className="grid-4">
                      <div className="stat-box">
                        <div className="stat-box-label">ELO Rating</div>
                        <div className="stat-box-value"
                          style={{ color }}>
                          {Math.round(r.elo)}
                        </div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-box-label">Win % vs Avg</div>
                        <div className="stat-box-value">
                          {winPct}%
                        </div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-box-label">
                          IPL Titles
                        </div>
                        <div className="stat-box-value">
                          {info.titles||0}
                          <span className="stat-box-unit">🏆</span>
                        </div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-box-label">Home City</div>
                        <div className="stat-box-value"
                          style={{ fontSize:16 }}>
                          {info.home||"—"}
                        </div>
                      </div>
                    </div>
                    <div style={{
                      marginTop:14, padding:"10px 14px",
                      background:"#111320",
                      borderRadius:8, fontSize:12,
                      color:"#9999bb", lineHeight:1.6,
                    }}>
                      <strong style={{ color:"#e8e8f0" }}>
                        ELO interpretation:
                      </strong>{" "}
                      {r.team} with ELO {Math.round(r.elo)} is
                      expected to win {winPct}% of matches against
                      a team with ELO 1500 (average).
                      {r.elo > 1550
                        ? " This is a historically strong team."
                        : r.elo < 1450
                        ? " This team has struggled historically."
                        : " This team is around average strength."}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        <div style={{ textAlign:"center", marginTop:16,
                      fontSize:11, color:"#444466" }}>
          Computed from 2008–2025 IPL data ·
          K-factor = 32 · Base ELO = 1500
        </div>
      </div>
    </div>
  )
}
