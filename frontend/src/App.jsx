import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/home";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<div style={{height:"100vh",display:"flex",justifyContent:"center",alignItems:"center"}}>Dashboard Coming Soon</div>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;