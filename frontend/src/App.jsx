import { useEffect, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_URL || ''

const CLASS_COLORS = {
  COVID: '#ef4444',
  Lung_Opacity: '#f59e0b',
  Normal: '#22c55e',
  'Viral Pneumonia': '#8b5cf6',
}

const HOW_IT_WORKS = [
  { icon: '📤', title: 'Upload', text: 'Your chest X-ray is resized to 224×224 and normalized with ImageNet statistics — the same preprocessing used during training.' },
  { icon: '🧠', title: 'Classify', text: 'A ResNet-18, fine-tuned on 21k+ radiographs with differential learning rates, scores the image across 4 classes (93.8% test accuracy).' },
  { icon: '📊', title: 'Probabilities', text: 'A softmax converts raw scores into the confidence distribution you see — the top class becomes the prediction.' },
  { icon: '🔥', title: 'Explain', text: 'Grad-CAM backpropagates the predicted class into the last conv block, highlighting the lung regions that drove the decision.' },
]

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [loading, setLoading] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState(null)
  const [apiUp, setApiUp] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((d) => setApiUp(d.model_loaded))
      .catch(() => setApiUp(false))
  }, [])

  const onFile = (f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    // data URL (not object URL) so it can be embedded in the downloaded HTML report
    const reader = new FileReader()
    reader.onload = () => setPreviewUrl(reader.result)
    reader.readAsDataURL(f)
    setResult(null)
    setHeatmap(null)
    setError(null)
  }

  const post = async (endpoint) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${API}${endpoint}`, { method: 'POST', body: form })
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({}))).error ?? (await res.json().catch(() => ({}))).detail
      throw new Error(detail || `Request failed (${res.status})`)
    }
    return res.json()
  }

  const predict = async () => {
    setLoading(true)
    setError(null)
    try {
      setResult(await post('/predict'))
      setHeatmap(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const explain = async () => {
    setExplaining(true)
    setError(null)
    try {
      const data = await post('/explain')
      setResult(data)
      setHeatmap(`data:image/png;base64,${data.heatmap_png_base64}`)
    } catch (e) {
      setError(e.message)
    } finally {
      setExplaining(false)
    }
  }

  const downloadReport = () => {
    if (!result) return
    const rows = Object.entries(result.probabilities)
      .sort((a, b) => b[1] - a[1])
      .map(
        ([name, p]) => `
        <tr>
          <td style="padding:6px 12px;color:#cbd5e1;">${name.replace('_', ' ')}</td>
          <td style="padding:6px 12px;">
            <div style="background:#1e293b;border-radius:99px;height:12px;width:260px;">
              <div style="height:100%;border-radius:99px;width:${Math.max(p * 100, 1)}%;background:${CLASS_COLORS[name] || '#3b82f6'};"></div>
            </div>
          </td>
          <td style="padding:6px 12px;color:#94a3b8;text-align:right;">${(p * 100).toFixed(1)}%</td>
        </tr>`
      )
      .join('')

    const heatmapBlock = heatmap
      ? `<div style="display:flex;gap:16px;margin-top:24px;">
           <figure style="margin:0;flex:1;"><img src="${previewUrl}" style="width:100%;border-radius:10px;"/><figcaption style="text-align:center;color:#64748b;font-size:12px;margin-top:6px;">Original</figcaption></figure>
           <figure style="margin:0;flex:1;"><img src="${heatmap}" style="width:100%;border-radius:10px;"/><figcaption style="text-align:center;color:#64748b;font-size:12px;margin-top:6px;">Grad-CAM heatmap</figcaption></figure>
         </div>
         <p style="color:#94a3b8;font-size:13px;line-height:1.6;">${result.explanation ?? ''}</p>`
      : `<div style="margin-top:24px;"><img src="${previewUrl}" style="max-width:320px;border-radius:10px;"/></div>`

    const html = `<!doctype html>
<html><head><meta charset="utf-8"/><title>LungLens Report</title></head>
<body style="background:#0f172a;font-family:Inter,system-ui,sans-serif;padding:40px;display:flex;justify-content:center;">
  <div style="background:#1e293bcc;border:1px solid #334155;border-radius:16px;padding:32px;max-width:720px;width:100%;">
    <h1 style="color:#e2e8f0;font-size:22px;margin:0 0 4px;">🫁 LungLens — Prediction Report</h1>
    <p style="color:#64748b;font-size:12px;margin:0 0 20px;">Generated ${new Date().toLocaleString()} · inference ${result.inference_time_ms} ms · research demo, not a medical device</p>
    <span style="background:${CLASS_COLORS[result.predicted_class] || '#3b82f6'};color:#fff;font-weight:700;font-size:17px;padding:8px 18px;border-radius:10px;">${result.predicted_class.replace('_', ' ')}</span>
    <span style="color:#94a3b8;font-size:14px;margin-left:12px;">${(result.confidence * 100).toFixed(1)}% confidence</span>
    <table style="border-collapse:collapse;margin-top:20px;">${rows}</table>
    ${heatmapBlock}
  </div>
</body></html>`

    const blob = new Blob([html], { type: 'text/html' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `lunglens_report_${result.predicted_class}.html`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="min-h-screen bg-slate-950 bg-[radial-gradient(circle_at_20%_0%,#1e293b_0%,#020617_60%)] text-slate-200">
      <div className="mx-auto max-w-7xl px-5 py-10">
        {/* Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-tight">
            🫁 Lung<span className="text-sky-400">Lens</span>
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Fine-tuned ResNet-18 chest X-ray classifier with Grad-CAM explainability
          </p>
          <div
            className={`mt-3 inline-block rounded-full bg-slate-800 px-4 py-1 text-xs ${
              apiUp ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {apiUp === null ? 'Checking API…' : apiUp ? '● Model loaded' : '● API / model unavailable'}
          </div>
        </header>

        {/* Two-column layout: upload + how-it-works LEFT, results RIGHT */}
        <main className="grid gap-6 lg:grid-cols-2">
          {/* LEFT column */}
          <div className="flex flex-col gap-6">
            <section
              className="flex min-h-[420px] cursor-pointer items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed border-slate-700 bg-slate-800/50 transition hover:border-sky-400 hover:bg-slate-800/80"
              onClick={() => inputRef.current.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                onFile(e.dataTransfer.files[0])
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => onFile(e.target.files[0])}
              />
              {previewUrl ? (
                <img src={previewUrl} alt="X-ray preview" className="max-h-[520px] w-full object-contain p-3" />
              ) : (
                <div className="p-8 text-center text-slate-400">
                  <div className="mb-3 text-4xl">📤</div>
                  <p className="text-lg">Drop a chest X-ray here, or click to browse</p>
                  <span className="text-xs text-slate-500">JPEG / PNG</span>
                </div>
              )}
            </section>

            <div className="flex justify-center gap-3">
              <button
                onClick={predict}
                disabled={!file || loading || explaining}
                className="rounded-xl bg-sky-500 px-8 py-3 font-semibold text-white transition hover:-translate-y-0.5 hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? 'Analyzing…' : 'Predict'}
              </button>
              <button
                onClick={explain}
                disabled={!file || loading || explaining}
                className="rounded-xl bg-violet-600 px-8 py-3 font-semibold text-white transition hover:-translate-y-0.5 hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {explaining ? 'Generating Grad-CAM…' : 'Predict + Explain'}
              </button>
            </div>

            {/* How this works */}
            <section className="rounded-2xl border border-slate-700 bg-slate-800/60 p-6">
              <h3 className="mb-4 text-base font-semibold text-slate-300">⚙️ How this works</h3>
              <ol className="space-y-4">
                {HOW_IT_WORKS.map((step, i) => (
                  <li key={step.title} className="flex gap-3">
                    <div className="text-2xl">{step.icon}</div>
                    <div>
                      <p className="text-sm font-semibold text-slate-200">
                        {i + 1}. {step.title}
                      </p>
                      <p className="text-xs leading-relaxed text-slate-400">{step.text}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          </div>

          {/* RIGHT column: results */}
          <div>
            {error && (
              <div className="mb-4 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                ⚠️ {error}
              </div>
            )}

            {result ? (
              <section className="rounded-2xl border border-slate-700 bg-slate-800/60 p-6">
                <div className="mb-5 flex flex-wrap items-center gap-3">
                  <span
                    className="rounded-xl px-5 py-2 text-lg font-bold text-white"
                    style={{ background: CLASS_COLORS[result.predicted_class] || '#3b82f6' }}
                  >
                    {result.predicted_class.replace('_', ' ')}
                  </span>
                  <span className="text-sm text-slate-400">
                    {(result.confidence * 100).toFixed(1)}% confidence · {result.inference_time_ms} ms
                  </span>
                  <button
                    onClick={downloadReport}
                    className="ml-auto rounded-lg border border-slate-600 bg-slate-700/60 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-600"
                  >
                    ⬇ Download report (.html)
                  </button>
                </div>

                <div className="space-y-2.5">
                  {sortedProbs.map(([name, p]) => (
                    <div key={name} className="grid grid-cols-[120px_1fr_52px] items-center gap-3 text-sm">
                      <span className="text-slate-300">{name.replace('_', ' ')}</span>
                      <div className="h-3 overflow-hidden rounded-full bg-slate-900">
                        <div
                          className="h-full rounded-full transition-[width] duration-500"
                          style={{
                            width: `${Math.max(p * 100, 1)}%`,
                            background: CLASS_COLORS[name] || '#3b82f6',
                          }}
                        />
                      </div>
                      <span className="text-right tabular-nums text-slate-400">{(p * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>

                {heatmap && (
                  <div className="mt-7 border-t border-slate-700 pt-5">
                    <h3 className="mb-4 text-base font-semibold text-slate-300">
                      Grad-CAM — where the model looked
                    </h3>
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                      <figure>
                        <img src={previewUrl} alt="Original X-ray" className="w-full rounded-xl" />
                        <figcaption className="mt-1.5 text-center text-xs text-slate-500">Original</figcaption>
                      </figure>
                      <figure>
                        <img src={heatmap} alt="Grad-CAM heatmap overlay" className="w-full rounded-xl" />
                        <figcaption className="mt-1.5 text-center text-xs text-slate-500">Grad-CAM heatmap</figcaption>
                      </figure>
                    </div>
                    <p className="mt-4 text-xs leading-relaxed text-slate-400">{result.explanation}</p>
                  </div>
                )}
              </section>
            ) : (
              <section className="flex h-full min-h-[420px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-900/40 p-6 text-center">
                <div className="text-slate-500">
                  <div className="mb-3 text-4xl">🩻</div>
                  <p className="text-sm">
                    Results will appear here.
                    <br />
                    Upload an X-ray and hit <span className="text-sky-400">Predict + Explain</span>.
                  </p>
                </div>
              </section>
            )}
          </div>
        </main>

        <footer className="mt-10 text-center text-xs text-slate-600">
          Research demo — not a medical device. Built with PyTorch Lightning, FastAPI &amp; React.
        </footer>
      </div>
    </div>
  )
}
