import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <div style={{
      height: "100vh",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      flexDirection: "column",
      backgroundColor: "#0f172a",
      color: "white"
    }}>
      <h1 style={{ fontSize: "40px", marginBottom: "20px" }}>
        ChainFlow Trading Dashboard
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