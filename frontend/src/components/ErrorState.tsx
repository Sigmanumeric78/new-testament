interface ErrorStateProps {
  message: string
}

export default function ErrorState({ message }: ErrorStateProps) {
  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700 shadow-sm" role="alert">
      <p className="font-semibold">Request failed</p>
      <p className="mt-1">{message}</p>
    </div>
  )
}
