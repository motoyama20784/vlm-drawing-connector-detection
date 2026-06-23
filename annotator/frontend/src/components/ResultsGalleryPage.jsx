import { useState, useEffect } from 'react'
import { fetchDirs, fetchResultsList, fetchImageUrl } from '../api.js'

function ResultBadge({ hasGt, resultFiles, metrics }) {
  if (!hasGt) {
    return (
      <span style={{
        padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
        background: '#2a1a1a', color: '#7a4a4a',
      }}>GT なし</span>
    )
  }
  if (!resultFiles || resultFiles.length === 0) {
    return (
      <span style={{
        padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
        background: '#1e3a5f', color: '#7a9cc0',
      }}>結果なし</span>
    )
  }
  if (!metrics) {
    return (
      <span style={{
        padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
        background: '#1e3a5f', color: '#7a9cc0',
      }}>評価不可</span>
    )
  }
  const { f1, tp, fp, fn } = metrics
  const bg = f1 >= 0.8 ? '#0d3320' : f1 >= 0.5 ? '#2a1f00' : '#2a0d0d'
  const color = f1 >= 0.8 ? '#00e676' : f1 >= 0.5 ? '#ffa726' : '#f44336'
  return (
    <span style={{
      padding: '2px 8px', borderRadius: '10px', fontSize: '11px', fontWeight: 'bold',
      background: bg, color,
    }}>
      F1:{f1.toFixed(2)}  TP:{tp} FP:{fp} FN:{fn}
    </span>
  )
}

export default function ResultsGalleryPage({ onSelectImage }) {
  const [dirs, setDirs] = useState([])
  const [selectedDir, setSelectedDir] = useState('')
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchDirs().then(d => {
      const srcDirs = d.filter(dir => !dir.endsWith('_masked'))
      setDirs(srcDirs)
      if (srcDirs.length > 0) setSelectedDir(srcDirs[0])
    }).catch(console.error)
  }, [])

  useEffect(() => {
    if (!selectedDir) return
    setLoading(true)
    fetchResultsList(selectedDir)
      .then(data => setImages(data.images))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [selectedDir])

  const withResults = images.filter(i => i.result_files?.length > 0).length

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0d1b2a', minHeight: 0 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: '16px', flexShrink: 0,
        padding: '8px 16px', background: '#0f1f10', borderBottom: '1px solid #1a3a1f',
      }}>
        <span style={{
          fontWeight: 'bold', fontSize: '15px', color: '#a8e6a8',
          borderLeft: '3px solid #4caf50', paddingLeft: '10px',
        }}>
          結果一覧
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontSize: '13px', color: '#6a9c6a' }}>ディレクトリ:</label>
          <select
            value={selectedDir}
            onChange={e => setSelectedDir(e.target.value)}
            style={{
              padding: '4px 8px', background: '#0f2015', color: '#e2eaf5',
              border: '1px solid #2a5a2e', borderRadius: '4px', fontSize: '13px',
            }}
          >
            {dirs.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        {images.length > 0 && (
          <span style={{ fontSize: '13px', color: '#6a9c6a' }}>
            {withResults} / {images.length} 件に結果あり
          </span>
        )}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        {loading ? (
          <div style={{ color: '#6a9c6a', padding: '32px', textAlign: 'center' }}>読み込み中...</div>
        ) : images.length === 0 ? (
          <div style={{ color: '#3a6a3a', padding: '32px', textAlign: 'center' }}>画像が見つかりません</div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))',
            gap: '16px',
          }}>
            {images.map(img => {
              const hasResult = img.result_files?.length > 0
              return (
                <div
                  key={img.filename}
                  onClick={() => hasResult && onSelectImage(img.filename, selectedDir)}
                  style={{
                    background: '#0f2015', border: '1px solid #1a3a1f',
                    borderRadius: '8px', overflow: 'hidden',
                    cursor: hasResult ? 'pointer' : 'default',
                    transition: 'border-color 0.15s, transform 0.15s',
                    opacity: hasResult ? 1 : 0.45,
                  }}
                  onMouseEnter={e => {
                    if (!hasResult) return
                    e.currentTarget.style.borderColor = '#4caf50'
                    e.currentTarget.style.transform = 'translateY(-2px)'
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = '#1a3a1f'
                    e.currentTarget.style.transform = 'translateY(0)'
                  }}
                >
                  <div style={{ width: '100%', height: '180px', overflow: 'hidden', background: '#0a1810' }}>
                    <img
                      src={fetchImageUrl(img.filename, selectedDir)}
                      alt={img.filename}
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      loading="lazy"
                    />
                  </div>
                  <div style={{ padding: '8px' }}>
                    <div style={{
                      fontSize: '11px', color: '#6a9c6a', marginBottom: '4px',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {img.filename}
                    </div>
                    <ResultBadge hasGt={img.has_gt} resultFiles={img.result_files} metrics={img.latest_metrics} />
                    {img.result_files?.length > 1 && (
                      <div style={{ fontSize: '10px', color: '#3a6a3a', marginTop: '3px' }}>
                        {img.result_files.length} 件の実験結果
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
