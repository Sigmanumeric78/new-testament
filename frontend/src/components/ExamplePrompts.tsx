const EXAMPLES = [
  'I am 75 kg male, fed, I drank 200 ml vodka in 1 hour. Should I keep drinking?',
  'Can I drive after drinking 180ml whisky?',
  'I am 60kg female and fasted, how drunk will I get after 180ml whisky?',
  'My friend is vomiting repeatedly and cannot wake up after drinking. What should I do?',
  'Why does wine give me headaches?',
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
          className="focus-ring w-full rounded-lg border border-slate-200 bg-white p-3 text-left text-xs text-slate-700 transition hover:border-brand-500"
        >
          {prompt}
        </button>
      ))}
    </div>
  )
}
