export default function VirtualButton({ onClick, label }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '20px 40px',
        fontSize: '24px',
        borderRadius: '12px',
        background: '#007bff',
        color: 'white',
        border: 'none',
        cursor: 'pointer',
        transition: 'transform 0.2s ease',
      }}
    >
      {label}
    </button>
  );
}
