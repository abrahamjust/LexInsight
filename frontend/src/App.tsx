import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import SearchPage from "./pages/Search";
import CasePage from "./pages/Case";
import BrowsePage from "./pages/Browse";
import EvaluationPage from "./pages/Evaluation";
import LabelsPage from "./pages/Labels";
import MethodPage from "./pages/Method";

function ThemeToggle() {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem("lexinsight-theme") ?? "auto";
    } catch {
      return "auto";
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("lexinsight-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return (
    <select className="theme-toggle" value={theme} onChange={(e) => setTheme(e.target.value)} aria-label="Theme">
      <option value="auto">System theme</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
  );
}

export default function App() {
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <Link to="/" className="brand">
            <span className="brand-mark">L</span>
            <span>
              LexInsight <small>Indian Supreme Court precedent retrieval</small>
            </span>
          </Link>
          <nav className="nav">
            <NavLink to="/" end>Search</NavLink>
            <NavLink to="/browse">Judgments</NavLink>
            <NavLink to="/evaluation">Evaluation</NavLink>
            <NavLink to="/labels">Relevance labels</NavLink>
            <NavLink to="/method">Method</NavLink>
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/case/:id" element={<CasePage />} />
          <Route path="/browse" element={<BrowsePage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/labels" element={<LabelsPage />} />
          <Route path="/method" element={<MethodPage />} />
        </Routes>
      </main>
    </>
  );
}
