import { useState } from 'react'
import { UploadCard } from '@/components/UploadCard'
import { ResultsPanel, type ClassPrediction } from '@/components/ResultsPanel'
import { GradCamPanel } from '@/components/GradCamPanel'
import { Activity } from 'lucide-react'

interface PredictResponse {
  predictions: Record<string, ClassPrediction>
  gradcam: {
    class: string
    image_base64: string
  }
}

function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleFileSelected(file: File) {
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setIsLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/predict', { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`server returned ${response.status}`)
      const data: PredictResponse = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#05060a] text-white">
      <div
        className="pointer-events-none fixed inset-0 -z-10"
        style={{
          background:
            'radial-gradient(circle at 20% 0%, rgba(56,189,248,0.15), transparent 50%), radial-gradient(circle at 80% 100%, rgba(168,85,247,0.12), transparent 50%)',
        }}
      />

      <header className="mx-auto flex max-w-5xl items-center gap-3 px-6 pt-10">
        <Activity className="size-6 text-sky-400" />
        <div>
          <h1 className="text-lg font-semibold">chest x-ray classifier</h1>
          <p className="text-sm text-white/40">
            multi-label disease detection · resnet50 · grad-cam
          </p>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-10">
        <UploadCard
          onFileSelected={handleFileSelected}
          previewUrl={previewUrl}
          isLoading={isLoading}
        />

        {error && (
          <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </p>
        )}

        {result && (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <ResultsPanel predictions={result.predictions} />
            {previewUrl && (
              <GradCamPanel
                originalUrl={previewUrl}
                gradcamBase64={result.gradcam.image_base64}
                gradcamClass={result.gradcam.class}
              />
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
