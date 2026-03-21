import { useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";
import "../styles/home.css";

function Home() {
  const navigate = useNavigate();
  const vantaRef = useRef(null);
  const titleRef = useRef(null);
  const subtitleRef = useRef(null);

  useEffect(() => {
    if (vantaRef.current && window.VANTA) {
      window.VANTA.FOG({
        el: vantaRef.current,
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00
      });
    }
  }, []);

  useEffect(() => {
    if (titleRef.current) {
      const text = "Welcome to QuantSent!";
      titleRef.current.innerHTML = text.split('').map((char, i) => 
        `<span class="char" style="animation-delay: ${i*100}ms">${char === ' ' ? '&nbsp;' : char}</span>`
      ).join('');
    }
    
    if (subtitleRef.current) {
      const subtitle = "AI-Powered Trading Solutions";
      let index = 0;
      const interval = setInterval(() => {
        if (index <= subtitle.length) {
          subtitleRef.current.textContent = subtitle.slice(0, index);
          index++;
        } else {
          clearInterval(interval);
        }
      }, 60);
      return () => clearInterval(interval);
    }
  }, []);

  return (
    <div ref={vantaRef} className="home-container">
      <div className="content-wrapper">
        <h1 className="title">
          <span ref={titleRef} className="title-text"></span>
        </h1>
        <p ref={subtitleRef} className="subtitle"></p>
        <button 
          className="start-button" 
          onClick={() => navigate("/dashboard")}
        >
          <span className="button-text">Make Profit</span>
        </button>
      </div>
    </div>
  );
}

export default Home;
