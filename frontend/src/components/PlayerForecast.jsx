import { useState, useEffect, useRef } from "react"
import axios from "axios"
import {
  ComposedChart, Area, Line, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid
} from "recharts"

const METRICS = [
  { value:"runs",        label:"Runs",        unit:"runs",  icon:"🏏" },
  { value:"average",     label:"Batting Avg", unit:"",      icon:"📊" },
  { value:"strike_rate", label:"Strike Rate", unit:"",      icon:"⚡" },
  { value:"wickets",     label:"Wickets",     unit:"wkts",  icon:"🎯" },
  { value:"economy",     label:"Economy",     unit:"",      icon:"💨" },
]

const CustomTooltip = ({ active, payload, label, metric }) => {
  if (!active || !payload?.length) return null
  const isForecast = label?.includes("pred")
  return (
    <div style={{
      background:"#1a1d2e", border:"1px solid #2e3250",
      borderRadius:8, padding:"10px 14px", fontSize:12,
    }}>
      <p style={{ color:"#9999bb", marginBottom:4 }}>
        {isForecast ? "🔮 Forecast" : `📅 ${label}`}
      </p>
      {payload.map(p => (
        p.name === "value" && (
          <p key={p.name} style={{ color:"#e8e8f0",
                                    fontWeight:700, fontSize:16 }}>
            {typeof p.value==="number" ? p.value.toFixed(1) : p.value}
            &nbsp;
            <span style={{ fontSize:12, color:"#9999bb" }}>
              {METRICS.find(m=>m.value===metric)?.unit}
            </span>
          </p>
        )
      ))}
    </div>
  )
}

export default function PlayerForecast() {
  const [players,     setPlayers]     = useState([])
  const [search,      setSearch]      = useState("")
  const [player,      setPlayer]      = useState("")
  const [metric,      setMetric]      = useState("runs")
  const [result,      setResult]      = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)
  const [suggestions, setSuggestions] = useState([])
  const [showDrop,    setShowDrop]    = useState(false)
  const dropRef = useRef(null)

  useEffect(() => {
    axios.get("/api/players")
      .then(r => setPlayers(r.data.players||[]))
      .catch(()=>{})
  }, [])

  useEffect(() => {
    if (search.length < 2) { setSuggestions([]); return }
    const q = search.toLowerCase()
    setSuggestions(
      players.filter(p=>p.name.toLowerCase().includes(q)).slice(0,8)
    )
  }, [search, players])

  async function forecast() {
    if (!player) { setError("Select a player first"); return }
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await axios.post("/api/predict/player",
        { player_name:player, metric })
      setResult(r.data)
    } catch(e) {
      setError(e.response?.data?.detail||
               "Forecast failed — is the backend running?")
    } finally { setLoading(false) }
  }

  const chartData = result ? [
    ...(result.historical||[]).map(h=>({
      season: String(h.season),
      value:  typeof h.value==="number" ? h.value : null,
    })),
    {
      season: "2026 (pred)",
      value:  result.forecast?.predicted,
      upper:  result.forecast?.upper_bound,
      lower:  result.forecast?.lower_bound,
      isForecast: true,
    }
  ] : []

  const metricObj = METRICS.find(m=>m.value===metric)

  return (
    <div>
      {/* Search card */}
      <div className="card">
        <div className="card-title">🔍 Find a Player</div>

        <div style={{ display:"grid",
                      gridTemplateColumns:"2fr 1fr",
                      gap:12, marginBottom:20 }}>
          {/* Player search */}
          <div style={{ position:"relative" }}>
            <label>Player Name</label>
            <input type="text"
              value={search}
              placeholder="Search by name… (e.g. Kohli, Bumrah)"
              onChange={e=>{
                setSearch(e.target.value)
                setPlayer(""); setResult(null)
                setShowDrop(true)
              }}
              onFocus={() => setShowDrop(true)}
            />
            {showDrop && suggestions.length>0 && (
              <div ref={dropRef} style={{
                position:"absolute", top:"100%", left:0, right:0,
                background:"#1a1d2e", border:"1px solid #2e3250",
                borderRadius:8, zIndex:100,
                maxHeight:260, overflowY:"auto",
                boxShadow:"0 8px 24px rgba(0,0,0,0.4)",
              }}>
                {suggestions.map(p=>(
                  <div key={p.name}
                    onClick={()=>{
                      setPlayer(p.name); setSearch(p.name)
                      setSuggestions([]); setShowDrop(false)
                      setResult(null)
                    }}
                    style={{
                      padding:"10px 14px", cursor:"pointer",
                      borderBottom:"1px solid #1a1d2e",
                      display:"flex", justifyContent:"space-between",
                      alignItems:"center",
                      background:"#111320",
                      transition:"background .1s",
                    }}
                    onMouseEnter={e=>e.currentTarget.style.background="#1a1d2e"}
                    onMouseLeave={e=>e.currentTarget.style.background="#111320"}
                  >
                    <span style={{ color:"#e8e8f0",
                                   fontWeight:500 }}>{p.name}</span>
                    <span style={{ fontSize:11, color:"#666688" }}>
                      {p.career_runs>0
                        ? `${p.career_runs.toLocaleString()} runs`
                        : `${p.career_wickets} wkts`}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Metric */}
          <div>
            <label>Stat to Forecast</label>
            <select value={metric}
              onChange={e=>{setMetric(e.target.value);setResult(null)}}>
              {METRICS.map(m=>(
                <option key={m.value} value={m.value}>
                  {m.icon} {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Metric quick-select pills */}
        <div style={{ display:"flex", gap:8,
                      flexWrap:"wrap", marginBottom:20 }}>
          {METRICS.map(m=>(
            <button key={m.value}
              className={`btn-sm ${metric===m.value
                ? "btn-sm--active":""}`}
              onClick={()=>{setMetric(m.value);setResult(null)}}>
              {m.icon} {m.label}
            </button>
          ))}
        </div>

        <button className="btn-primary" onClick={forecast}
          disabled={loading||!player}>
          {loading
            ? <><div className="spinner"/>Forecasting…</>
            : `📈 Forecast ${player||"Player"} — ${metricObj?.label}`}
        </button>

        {error && <div className="alert-error">{error}</div>}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Forecast headline */}
          <div className="card" style={{ marginTop:16 }}>
            <div className="card-title">
              🔮 2026 Forecast — {result.player}
            </div>

            {/* Big predicted number */}
            <div style={{ textAlign:"center", padding:"20px 0" }}>
              <div style={{ fontSize:11, color:"#9999bb",
                            textTransform:"uppercase",
                            letterSpacing:1, marginBottom:8 }}>
                Predicted {metricObj?.label} · Next Season
              </div>
              <div style={{ fontSize:56, fontWeight:800,
                            color:"#4f7bef",
                            lineHeight:1 }}>
                {result.forecast?.predicted?.toFixed(
                  result.forecast?.predicted > 10 ? 0 : 1
                )}
                <span style={{ fontSize:18, color:"#9999bb",
                               marginLeft:8 }}>
                  {metricObj?.unit}
                </span>
              </div>
              <div style={{ fontSize:13, color:"#666688",
                            marginTop:8 }}>
                80% CI: {result.forecast?.lower_bound?.toFixed(0)}
                &nbsp;–&nbsp;
                {result.forecast?.upper_bound?.toFixed(0)}
                &nbsp;&nbsp;·&nbsp;&nbsp;
                {result.forecast?.seasons_of_data} seasons of data
                &nbsp;&nbsp;·&nbsp;&nbsp;
                method: {result.forecast?.method}
              </div>
            </div>

            {/* Stat boxes */}
            <div className="grid-4" style={{ marginBottom:20 }}>
              <div className="stat-box">
                <div className="stat-box-label">Predicted</div>
                <div className="stat-box-value">
                  {result.forecast?.predicted?.toFixed(
                    result.forecast.predicted>10?0:1)}
                  <span className="stat-box-unit">
                    {metricObj?.unit}
                  </span>
                </div>
              </div>
              <div className="stat-box">
                <div className="stat-box-label">Lower (80% CI)</div>
                <div className="stat-box-value">
                  {result.forecast?.lower_bound?.toFixed(0)}
                </div>
              </div>
              <div className="stat-box">
                <div className="stat-box-label">Upper (80% CI)</div>
                <div className="stat-box-value">
                  {result.forecast?.upper_bound?.toFixed(0)}
                </div>
              </div>
              <div className="stat-box">
                <div className="stat-box-label">Data Points</div>
                <div className="stat-box-value">
                  {result.forecast?.seasons_of_data}
                  <span className="stat-box-unit">seasons</span>
                </div>
              </div>
            </div>

            {/* Chart */}
            {chartData.length > 1 && (
              <div>
                <div style={{ fontSize:12, color:"#666688",
                              marginBottom:12 }}>
                  Career trend + 2026 forecast with 80%
                  confidence interval
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <ComposedChart data={chartData}
                    margin={{left:0,right:16,top:8,bottom:0}}>
                    <CartesianGrid strokeDasharray="3 3"
                      stroke="#1a1d2e" vertical={false}/>
                    <XAxis dataKey="season"
                      tick={{fill:"#666688",fontSize:11}}
                      axisLine={{stroke:"#252840"}}
                      tickLine={false}/>
                    <YAxis
                      tick={{fill:"#666688",fontSize:11}}
                      axisLine={false} tickLine={false}/>
                    <Tooltip
                      content={<CustomTooltip metric={metric}/>}/>
                    <Area dataKey="upper"
                      fill="rgba(79,123,239,0.08)"
                      stroke="none" name="upper"/>
                    <Area dataKey="lower"
                      fill="#0b0d14" stroke="none" name="lower"/>
                    <Line dataKey="value"
                      stroke="#4f7bef" strokeWidth={2.5}
                      dot={(props)=>{
                        const {cx,cy,payload}=props
                        return <circle key={cx}
                          cx={cx} cy={cy} r={5}
                          fill={payload.isForecast
                            ?"#f59e0b":"#4f7bef"}
                          stroke={payload.isForecast
                            ?"#1a1d2e":"#1a1d2e"}
                          strokeWidth={2}/>
                      }}
                    />
                    <ReferenceLine x="2026 (pred)"
                      stroke="#f59e0b" strokeDasharray="4 4"/>
                  </ComposedChart>
                </ResponsiveContainer>
                <div style={{ display:"flex", gap:16,
                              justifyContent:"center",
                              marginTop:8, fontSize:11,
                              color:"#666688" }}>
                  <span>
                    <span style={{ color:"#4f7bef" }}>●</span>
                    &nbsp;Historical
                  </span>
                  <span>
                    <span style={{ color:"#f59e0b" }}>●</span>
                    &nbsp;Forecast
                  </span>
                  <span>
                    <span style={{ color:"rgba(79,123,239,0.4)" }}>
                      ▬▬
                    </span>
                    &nbsp;80% CI band
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Career stats */}
          {result.career_stats && (
            <div className="card" style={{ marginTop:16 }}>
              <div className="card-title">
                📋 Career Statistics
              </div>
              <div className="grid-4">
                {result.career_stats.career_runs>0 && (
                  <div className="stat-box">
                    <div className="stat-box-label">Career Runs</div>
                    <div className="stat-box-value">
                      {result.career_stats.career_runs
                        .toLocaleString()}
                    </div>
                  </div>
                )}
                {result.career_stats.career_avg>0 && (
                  <div className="stat-box">
                    <div className="stat-box-label">Batting Avg</div>
                    <div className="stat-box-value">
                      {result.career_stats.career_avg.toFixed(1)}
                    </div>
                  </div>
                )}
                {result.career_stats.career_sr>0 && (
                  <div className="stat-box">
                    <div className="stat-box-label">Strike Rate</div>
                    <div className="stat-box-value">
                      {result.career_stats.career_sr.toFixed(1)}
                    </div>
                  </div>
                )}
                {result.career_stats.career_wickets>0 && (
                  <div className="stat-box">
                    <div className="stat-box-label">
                      Career Wickets
                    </div>
                    <div className="stat-box-value">
                      {result.career_stats.career_wickets}
                    </div>
                  </div>
                )}
                {result.career_stats.career_economy>0 && (
                  <div className="stat-box">
                    <div className="stat-box-label">Economy</div>
                    <div className="stat-box-value">
                      {result.career_stats.career_economy.toFixed(2)}
                    </div>
                  </div>
                )}
              </div>
              <p style={{ fontSize:11, color:"#444466",
                           marginTop:12 }}>
                {result.disclaimer}
              </p>
            </div>
          )}
        </>
      )}

      {/* Top players quick list */}
      {!result && players.length>0 && (
        <div className="card" style={{ marginTop:16 }}>
          <div className="card-title">⭐ Top Players Available</div>
          <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
            {players.slice(0,20).map(p=>(
              <button key={p.name} className="btn-sm"
                onClick={()=>{
                  setPlayer(p.name); setSearch(p.name)
                  setSuggestions([]); setShowDrop(false)
                }}>
                {p.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
