const EXAMPLES = [
  'Is this product high in sodium or added sugar?',
  'Explain maltodextrin in simple terms.',
  'Compare oats and cornflakes for weight management.',
  'I am 75 kg male, fed, and drank 50 ml vodka in 1 hour. What is my estimated risk?',
  'Why can some wines cause headaches?',
]

interface ExamplePromptsProps {
  onSelect: (prompt: string) => void
}

export default function ExamplePrompts({ onSelect }: ExamplePromptsProps) {
  return (
    <div className="space-y-2">
      {EXAMPLES.map((prompt) => (
        <button
          key={prompt}
          type="button"
          onClick={() => onSelect(prompt)}
          className="focus-ring w-full rounded-lg border border-slate-200 bg-white p-3 text-left text-xs text-slate-700 transition hover:border-brand-500 hover:bg-emerald-50"
        >
          {prompt}
        </button>
      ))}
    </div>
  )
}
