export default function BboxList({ bboxes, onCategoryChange, onInfer, onDelete, inferring }) {
  if (bboxes.length === 0) {
    return (
      <div style={{ padding: '16px', color: '#666', fontSize: '13px' }}>
        Canvas 上でドラッグして bbox を描いてください
      </div>
    )
  }

  return (
    <div style={{ overflowY: 'auto', flex: 1 }}>
      {bboxes.map((bbox, i) => (
        <div key={bbox.id} style={{
          padding: '10px',
          borderBottom: '1px solid #333',
          background: '#2a2a2a',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontWeight: 'bold', color: '#00e676' }}>bbox-{i + 1}</span>
            <button
              onClick={() => onDelete(bbox.id)}
              style={{ background: 'none', border: 'none', color: '#f44336', cursor: 'pointer', fontSize: '16px' }}
            >×</button>
          </div>

          <div style={{ marginBottom: '6px' }}>
            <label style={{ fontSize: '12px', color: '#aaa' }}>カテゴリ</label>
            <input
              type="text"
              value={bbox.category}
              onChange={e => onCategoryChange(bbox.id, e.target.value)}
              placeholder="例: terminal"
              style={{
                width: '100%', marginTop: '2px', padding: '4px 6px',
                background: '#333', color: '#eee', border: '1px solid #555', borderRadius: '4px',
                fontSize: '13px',
              }}
            />
          </div>

          <button
            onClick={() => onInfer(bbox)}
            disabled={inferring === bbox.id}
            style={{
              width: '100%', padding: '5px',
              background: inferring === bbox.id ? '#555' : '#1565c0',
              color: '#fff', border: 'none', borderRadius: '4px',
              cursor: inferring === bbox.id ? 'not-allowed' : 'pointer',
              fontSize: '13px', marginBottom: '6px',
            }}
          >
            {inferring === bbox.id ? '推論中...' : 'VLM 推論'}
          </button>

          {(bbox.vlm_text !== null || bbox.vlm_shape !== null) && (
            <div style={{ fontSize: '12px', color: '#bbb', background: '#1a1a1a', padding: '6px', borderRadius: '4px' }}>
              <div>テキスト: <span style={{ color: '#fff' }}>{bbox.vlm_text ?? '—'}</span></div>
              <div>形状: <span style={{ color: '#fff' }}>{bbox.vlm_shape ?? '—'}</span></div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
