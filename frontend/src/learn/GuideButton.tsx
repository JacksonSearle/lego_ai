import { useLearn } from './LearnContext'

/** The “?” button on each page: opens that page’s guidebook lesson. */
export function GuideButton({ lesson, label }: { lesson: string; label?: string }) {
  const { openLesson } = useLearn()
  return (
    <button className="ghost sm guide-btn" title="Open the guide for this page" onClick={() => openLesson(lesson)}>
      ? {label}
    </button>
  )
}
