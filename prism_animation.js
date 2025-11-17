
const PRISM_PROPS = {
    height: 3.5,
    basewidth: 5.5,
    animationType: "rotate", 
    glow: 1,
    offset: { x: 0, y: 0 },
    noise: 0.5,
    transparent: true
};

const canvas = document.getElementById('prism-bg');
const ctx = canvas.getContext('3d');
let rotationAngle = 0;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function drawPrismSimulation() {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2 + PRISM_PROPS.offset.x;
    const centerY = canvas.height / 2 + PRISM_PROPS.offset.y;
    
    // --- Simulation of Glow/Noise ---
    ctx.globalAlpha = 0.05 + PRISM_PROPS.noise * 0.1; // Use noise for opacity
    ctx.fillStyle = '#a78bfa'; // Purple glow color (from previous code)
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;

    // --- Simulation of Rotating Prism ---
    const scale = 50; 
    const baseRadius = PRISM_PROPS.basewidth * scale;
    const prismHeight = PRISM_PROPS.height * scale;

    ctx.save();
    ctx.translate(centerX, centerY);
    
    // Apply rotation based on animationType="rotate"
    rotationAngle += 0.005; 
    ctx.rotate(rotationAngle);

    // Draw the simulated prism base (a hexagon for a complex look)
    const sides = 6;
    ctx.beginPath();
    for (let i = 0; i < sides; i++) {
        const angle = i * 2 * Math.PI / sides;
        const x = baseRadius * Math.cos(angle);
        const y = baseRadius * 0.5 * Math.sin(angle); // Squish Y for 3D perspective
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.closePath();
    
    // Fill with a dynamic color and apply glow
    ctx.strokeStyle = `rgba(167, 139, 250, ${PRISM_PROPS.glow * 0.5})`;
    ctx.shadowColor = `rgba(167, 139, 250, ${PRISM_PROPS.glow})`;
    ctx.shadowBlur = 20;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw the 'apex'
    ctx.beginPath();
    ctx.arc(0, -prismHeight / 3, 5, 0, 2 * Math.PI);
    ctx.fillStyle = 'white';
    ctx.fill();

    ctx.restore();

    requestAnimationFrame(drawPrismSimulation);
}

// Start the animation loop when the window loads
window.addEventListener('load', () => {
    resizeCanvas();
    drawPrismSimulation();
});
window.addEventListener('resize', resizeCanvas);