export default function ImageSelector({ images, selected, onChange }) {
  return (
    <div style={{ padding: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
      <label style={{ fontSize: '14px', color: '#7a9cc0' }}>画像:</label>
      <select
        value={selected}
        onChange={e => onChange(e.target.value)}
        style={{ padding: '4px 8px', background: '#1b2d3e', color: '#e2eaf5', border: '1px solid #2a4060', borderRadius: '4px' }}
      >
        <option value="">-- 選択してください --</option>
        {images.map(img => (
          <option key={img} value={img}>{img}</option>
        ))}
      </select>
    </div>
  )
}
