import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

export interface ClassPrediction {
  probability: number
  predicted: boolean
}

interface ResultsPanelProps {
  predictions: Record<string, ClassPrediction>
}

export function ResultsPanel({ predictions }: ResultsPanelProps) {
  const entries = Object.entries(predictions).sort(
    (a, b) => b[1].probability - a[1].probability,
  )

  return (
    <Card className="border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl shadow-black/40">
      <CardHeader>
        <CardTitle className="text-white/90">predictions</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {entries.map(([className, { probability, predicted }], i) => (
          <div
            key={className}
            className="flex flex-col gap-1.5 animate-in fade-in slide-in-from-bottom-2"
            style={{ animationDelay: `${i * 60}ms`, animationDuration: '400ms', animationFillMode: 'both' }}
          >
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-white/80">{className}</span>
              <div className="flex items-center gap-2">
                <span className="text-white/50 tabular-nums">
                  {(probability * 100).toFixed(1)}%
                </span>
                {predicted && (
                  <Badge variant="default" className="bg-sky-500/90 text-white">
                    positive
                  </Badge>
                )}
              </div>
            </div>
            <Progress value={probability * 100} className="h-2 bg-white/10" />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
