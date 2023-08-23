import { faCheck } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

export function PlanInputSummaryChoiceItem<Value>({
  value,
  label,
  selected,
}: {
  value: Value
  label: (value: Value) => string
  selected: (value: Value) => boolean
}) {
  return (
    <h2 className={selected(value) ? '' : 'lighten-2'}>
      {label(value)}
      {selected(value) && <FontAwesomeIcon className="ml-2" icon={faCheck} />}
    </h2>
  )
}
