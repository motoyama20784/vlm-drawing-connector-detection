export default function BboxList({ bboxes, selectedId, onSelect, onCategoryChange, onInfer, onDelete, inferring }) {
  if (bboxes.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Canvas 上でドラッグして bbox を描いてください
      </div>
    )
  }

  return (
    <div style={{ overflowY: 'auto', flex: 1 }}>
      {bboxes.map((bbox, i) => {
        const isSelected = bbox.id === selectedId
        return (
          <div
            key={bbox.id}
            onClick={() => onSelect(bbox.id)}
            style={{
              padding: '10px',
              borderBottom: '1px solid #333',
              background: isSelected ? '#2a3a2a' : '#2a2a2a',
              borderLeft: isSelected ? '3px solid #ff9800' : '3px solid transparent',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontWeight: 'bold', color: isSelected ? '#ff9800' : '#00e676' }}>
                bbox-{i + 1}
              </span>
              <button
                onClick={e => { e.stopPropagation(); onDelete(bbox.id) }}
                style={{ background: 'none', border: 'none', color: '#f44336', cursor: 'pointer', fontSize: '16px' }}
              >×</button>
            </div>

            <div style={{ marginBottom: '6px' }} onClick={e => e.stopPropagation()}>
              <label style={{ fontSize: '12px', color: '#aaa' }}>カテゴリ</label>
              <input
                type="text"
                value={bbox.category}
                onChange={e => onCategoryChange(bbox.id, e.target.value)}
                placeholder="例: terminal"
                style={{
                  width: '100%', marginTop: '2px', padding: '4px 6px',
                  background: '#333', color: '#eee', border: `1px solid ${isSelected ? '#ff9800' : '#555'}`,
                  borderRadius: '4px', fontSize: '13px',
                }}
              />
            </div>

            <button
              onClick={e => { e.stopPropagation(); onInfer(bbox) }}
              disabled={inferring === bbox.id}
              style={{
                width: '100%', padding: '5px',
                background: inferring === bbox.id ? '#555' : '#1565c0',
                color: '#fff', border: 'none', borderRadius: '4px',
                cursor: inferring === bbox.id ? 'not-allowed' : 'pointer',
                fontSize: '13px', marginBottom: '6px',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              }}
            >
              {inferring === bbox.id && <span className="spinner" />}
              {inferring === bbox.id ? '推論中...' : 'VLM 推論'}
            </button>

            {(bbox.vlm_text !== null || bbox.vlm_shape !== null) && (
              <div style={{ fontSize: '12px', color: '#bbb', background: '#1a1a1a', padding: '6px', borderRadius: '4px' }}>
                <div>テキスト: <span style={{ color: '#fff' }}>{bbox.vlm_text ?? '—'}</span></div>
                <div>形状: <span style={{ color: '#fff' }}>{bbox.vlm_shape ?? '—'}</span></div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
