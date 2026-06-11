import { useState } from "react"
import MatchPredictor from "./components/MatchPredictor"
import PlayerForecast from "./components/PlayerForecast"
import TournamentSim  from "./components/TournamentSim"
import EloRankings    from "./components/EloRankings"
import "./App.css"

const TABS = [
  { id:"match",      label:"Match Predictor",     icon:"🏏",
    desc:"Predict any IPL matchup from 2025 to 2049" },
  { id:"player",     label:"Player Forecast",      icon:"📈",
    desc:"Next season stats predicted by Prophet ML" },
  { id:"tournament", label:"Tournament Simulator", icon:"🏆",
    desc:"Simulate a full IPL season and get predicted standings" },
  { id:"rankings",   label:"Power Rankings",       icon:"⚡",
    desc:"ELO-based team strength ratings from 18 seasons of data" },
]

export default function App() {
  const [tab, setTab] = useState("match")
  const active = TABS.find(t => t.id === tab)

  return (
    <div className="app">
      {/* HERO */}
      <header className="hero">
        <div className="hero-glow" />
        <div className="hero-content">
          <div className="hero-badge">ML-Powered · 2025–2049</div>
          <h1 className="hero-title">🏏 IPL Predictor</h1>
          <p className="hero-sub">
            Ball-by-ball phase features · Prophet player forecasting
            · ELO team ratings · Stacking ensemble
          </p>
          <div className="hero-pills">
            <span className="pill pill-blue">87 Players Forecast</span>
            <span className="pill pill-green">1076 Matches Trained</span>
            <span className="pill pill-orange">18 IPL Seasons</span>
            <span className="pill pill-purple">CV AUC 0.55</span>
          </div>
        </div>
      </header>

      {/* NAV */}
      <nav className="nav">
        <div className="nav-inner">
          {TABS.map(t => (
            <button key={t.id}
              onClick={() => setTab(t.id)}
              className={`nav-tab ${tab===t.id ? "nav-tab--active":""}`}>
              <span className="nav-tab-icon">{t.icon}</span>
              <span className="nav-tab-label">{t.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* PAGE HEADER */}
      <div className="page-hdr">
        <div className="page-hdr-inner">
          <h2 className="page-hdr-title">
            {active?.icon} {active?.label}
          </h2>
          <p className="page-hdr-sub">{active?.desc}</p>
        </div>
      </div>

      {/* CONTENT */}
      <main className="main">
        {tab==="match"      && <MatchPredictor />}
        {tab==="player"     && <PlayerForecast />}
        {tab==="tournament" && <TournamentSim  />}
        {tab==="rankings"   && <EloRankings    />}
      </main>

      {/* FOOTER */}
      <footer className="footer">
        IPL Predictor &nbsp;·&nbsp; FastAPI + React + XGBoost + Prophet
        &nbsp;·&nbsp; Built with real IPL data (2008–2025)
      </footer>
    </div>
  )
}
