/**
 * To run: 1) run backend server
 *         2) open new terminal at '.../frontend/src'
 *         3) run 'npm run dev'
 *         4) go to localhost:5173 
 */



import React, { useEffect, useState } from 'react';

import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  ScatterChart, Scatter, ZAxis, LineChart, Line, CartesianGrid 
} from 'recharts';

import './App.css'; //Connect to css file

function App() {
  const [planets, setPlanets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('rankings'); // State for switching tabs
  const [selectedPlanet, setSelectedPlanet] = useState(null);
  const [edaData, setEdaData] = useState({ timeline: [], scatter: [] });
  const [stats, setStats] = useState([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/top-planets')
      .then(res => res.json())
      .then(data => {
        setPlanets(data);
        setLoading(false);
      });
  }, []);

  return (
    <div className="app-container">
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ color: 'white', letterSpacing: '2px' }}>NASA Exoplanet Archive Data Analysis</h1>
        
        {/* TAB BUTTONS */}
        <div style={{ marginTop: '20px' }}>
          <button 
            className={`tab-button ${tab === 'rankings' ? 'active' : ''}`}
            onClick={() => setTab('rankings')}
          >
            Top Candidates
          </button>
          <button 
            className={`tab-button ${tab === 'eda' ? 'active' : ''}`}
            onClick={() => setTab('eda')}
          >
            Data Insights
          </button>
        </div>
      </header>

      {loading ? (
        <p style={{ textAlign: 'center' }}>Initializing Deep Space Sensors...</p>
      ) : tab === 'rankings' ? (
        /* --- RANKINGS TABLE TAB --- */
        <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
          <table className="planet-table">
            <thead>
              <tr style={{ color: '#58a6ff', textTransform: 'uppercase', fontSize: '0.8rem' }}>
                <th style={{ padding: '10px 20px', textAlign: 'left' }}>Candidate Name</th>
                <th style={{ padding: '10px 20px', textAlign: 'left' }}>Habitability Potential Prediction</th>
                <th style={{ padding: '10px 20px', textAlign: 'left' }}>Habitability Zone Prediction</th>
                <th style={{ padding: '10px 20px', textAlign: 'left' }}>ESI Prediction</th>
                <th style={{ padding: '10px 20px', textAlign: 'left' }}>PHL Comparison</th>
              </tr>
            </thead>
            <tbody>
              {planets.map((p, i) => (
                <tr key={i} className="planet-row" onClick={() => setSelectedPlanet(p)}>
                  <td className="cell cell-first">
                    <strong>{p.name}</strong>
                  </td>
                  <td className="cell" style={{ color: '#7ee787', fontWeight: 'bold' }}>
                    {(p.potential * 100).toFixed(1)}%
                  </td>
                  <td className="cell">{(p.hzProb * 100).toFixed(0)}%</td>
                  <td className="cell">{p.esi.toFixed(3)}</td>
                  <td className="cell cell-last">
                    {/*Check if the planet exists in the PHL dataset at all */}
                    {p.phl_status === "Not Listed" || p.phl_status == -1 ? (
                      <span style={{ color: '#8b949e', fontStyle: 'italic', fontSize: '0.85rem' }}>
                        Not in PHL Dataset
                      </span>
                    ) : (
                      /*If it does exist, check if it's Habitable (1) or not (0) */
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ 
                          fontSize: '1.2rem', 
                          color: (p.phl_status == 1) ? '#7ee787' : '#ff7b72' 
                        }}>
                          {p.phl_status == 1 ? "✔" : "✘"}
                        </span>
                        <span style={{ 
                          fontSize: '0.75rem', 
                          color: (p.phl_status == 1) ? '#7ee787' : '#ff7b72',
                          textTransform: 'uppercase',
                          letterSpacing: '1px'
                        }}>
                          {p.phl_status == 1 ? "Confirmed" : "Refuted"}
                        </span>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="eda-container" style={{ maxWidth: '1100px', margin: '0 auto', color: 'white' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '30px' }}>
            
            {/* CHART 1: Feature Importance*/}
            <div className="stats-card">
              <h4>Model Decision Drivers</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stats} layout="vertical">
                  <XAxis type="number" hide />
                  <YAxis dataKey="name" type="category" stroke="#8b949e" fontSize={10} width={80} />
                  <Tooltip contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #58a6ff' }} />
                  <Bar dataKey="value" fill="#58a6ff" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* CHART 2: Discovery Timeline */}
            <div className="stats-card">
              <h4>Discoveries Over Time</h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={edaData.timeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
                  <XAxis dataKey="year" stroke="#8b949e" fontSize={10} />
                  <YAxis stroke="#8b949e" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #58a6ff' }} />
                  <Line type="monotone" dataKey="count" stroke="#7ee787" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* CHART 3: The "Habitable Strip" Scatter Plot */}
          <div className="stats-card" style={{ height: '400px' }}>
            <h4>Planet Radius vs. Host Star Temperature</h4>
            <p style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: '20px' }}>
              Small planets orbiting cooler stars (bottom left) are the primary targets for habitability.
            </p>
            <ResponsiveContainer width="100%" height="80%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid stroke="#30363d" />
                <XAxis type="number" dataKey="st_teff" name="Star Temp" unit="K" stroke="#8b949e" domain={['auto', 'auto']} />
                <YAxis type="number" dataKey="pl_rade" name="Planet Radius" unit=" R⊕" stroke="#8b949e" domain={[0, 20]} />
                <ZAxis type="category" dataKey="pl_name" name="Planet" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0d1117', border: '1px solid #58a6ff' }} />
                <Scatter name="Planets" data={edaData.scatter} fill="#58a6ff" opacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {selectedPlanet && (
        <div className="modal-overlay" onClick={() => setSelectedPlanet(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setSelectedPlanet(null)}>×</button>
            
            <h2 style={{ color: '#58a6ff' }}>{selectedPlanet.name}</h2>
            <hr style={{ borderColor: '#30363d', margin: '20px 0' }} />
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', textAlign: 'left' }}>
              <div>
                <p style={{ color: '#8b949e' }}>Habitability Probability</p>
                <h3 style={{ color: '#7ee787' }}>{(selectedPlanet.hzProb * 100).toFixed(1)}%</h3>
              </div>
              <div>
                <p style={{ color: '#8b949e' }}>Earth Similarity Index</p>
                <h3>{selectedPlanet.esi.toFixed(3)}</h3>
              </div>
            </div>

            <div style={{ marginTop: '30px', padding: '15px', backgroundColor: '#0d1117', borderRadius: '8px', border: '1px solid #30363d' }}>
              <h4 style={{ color: '#58a6ff', marginTop: 0 }}>Orbital Dynamics</h4>
              <p style={{ fontSize: '0.9rem', lineHeight: '1.6' }}>
                This planet maintains a stable orbit with a relative flux of <strong>{selectedPlanet.potential.toFixed(4)}</strong>. 
                Its position relative to the host star suggests a high likelihood of liquid water retention based on our 2-stage ML model.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;