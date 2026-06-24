import { useCallback, useRef, useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { UploadCloud, ImageIcon } from 'lucide-react'

interface UploadCardProps {
  onFileSelected: (file: File) => void
  previewUrl: string | null
  isLoading: boolean
}

export function UploadCard({ onFileSelected, previewUrl, isLoading }: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files?.[0]
      if (file) onFileSelected(file)
    },
    [onFileSelected],
  )

  return (
    <Card className="border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl shadow-black/40">
      <CardContent>
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer ${
            isDragging
              ? 'border-sky-400 bg-sky-400/10'
              : 'border-white/15 hover:border-white/30 hover:bg-white/5'
          }`}
        >
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="upload preview"
              className="max-h-64 rounded-lg object-contain"
            />
          ) : (
            <>
              <UploadCloud className="size-10 text-white/40" />
              <p className="text-sm text-white/70">
                drag and drop a chest x-ray, or click to browse
              </p>
              <p className="flex items-center gap-1 text-xs text-white/40">
                <ImageIcon className="size-3" /> PNG or JPG
              </p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onFileSelected(file)
            }}
          />
        </div>
        {previewUrl && (
          <Button
            variant="secondary"
            className="mt-4 w-full"
            disabled={isLoading}
            onClick={() => inputRef.current?.click()}
          >
            {isLoading ? 'analyzing...' : 'choose a different image'}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
