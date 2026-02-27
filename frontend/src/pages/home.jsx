import { useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";

function Home() {
  const navigate = useNavigate();
  const vantaRef = useRef(null);

  useEffect(() => {
    if (vantaRef.current && window.VANTA) {
      window.VANTA.WAVES({
        el: vantaRef.current,
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.00,
        scaleMobile: 1.00,
        color: 0xf0f27,
        shininess: 43.00,
        waveSpeed: 1.05,
        zoom: 1.25
      });
    }
  }, []);

  return (
    <div ref={vantaRef} style={{
      height: "100vh",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      flexDirection: "column",
      color: "white"
    }}>
      <h1 style={{ fontSize: "40px", marginBottom: "50px" }}>
        ALGO TRADING
      </h1>

      <button
        onClick={() => navigate("/dashboard")}
        style={{
          padding: "12px 24px",
          fontSize: "18px",
          cursor: "pointer",
          backgroundColor: "#22c55e",
          border: "none",
          borderRadius: "8px"
        }}
      >
        Start
      </button>
    </div>
  );
}

export default Home;