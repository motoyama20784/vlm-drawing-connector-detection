export default function ImageSelector({ images, selected, onChange }) {
  return (
    <div style={{ padding: '8px', background: '#2a2a2a', display: 'flex', alignItems: 'center', gap: '8px' }}>
      <label style={{ fontSize: '14px' }}>画像:</label>
      <select
        value={selected}
        onChange={e => onChange(e.target.value)}
        style={{ padding: '4px 8px', background: '#333', color: '#eee', border: '1px solid #555', borderRadius: '4px' }}
      >
        <option value="">-- 選択してください --</option>
        {images.map(img => (
          <option key={img} value={img}>{img}</option>
        ))}
      </select>
    </div>
  )
}
