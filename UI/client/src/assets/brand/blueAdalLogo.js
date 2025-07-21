const BlueAdalLogo = ({ bg = 'none', stroke = '#1E90FF', width = 180, height = 100, text = 'BlueAdal', textColor = '#ffffff', fontSize = 32, ...props }) => (
  <svg width={width} height={height} viewBox="0 0 180 100" {...props} 
      style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'flex-start', border: '2px' }}>
    {/* Logo on the far left, removed translate for minimal left padding */}
    <g>
      <circle cx={45} cy={50} r={45} fill={bg} stroke={stroke} strokeWidth={2} />
      <path
        d="M45 20 A10 10 0 0 1 45 40 M35 30 L25 50 M55 30 L65 50 M25 50 L35 70 M65 50 L55 70"
        stroke="#000000"
        strokeWidth={3}
        fill="none"
      />
      <path d="M15 50 H25 M65 50 H75" stroke={stroke} strokeWidth={2} />
      <circle cx={45} cy={20} r={3} fill={stroke} />
    </g>
    {/* Text positioned just to the right of the logo */}
    <text x="95" y="60" textAnchor="start" fontFamily="Segoe UI, Arial, sans-serif" fontSize={fontSize} fontWeight="bold" fill={textColor}>{text}</text>
  </svg>
);

export default BlueAdalLogo;