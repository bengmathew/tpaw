import React, {
  ReactElement,
  ReactNode,
  useEffect,
  useLayoutEffect,
  useState,
} from 'react'
import { newPaddingHorz, paddingCSSStyleHorz } from '../../../../Utils/Geometry'
import { fGet } from '../../../../Utils/Utils'
import { useSimulation } from '../../../App/WithSimulation'
import { ModalBase } from '../../../Common/Modal/ModalBase'
import { PlanInputType } from '../Helpers/PlanInputType'
import { PlanInputSizing } from '../PlanInput'
import { PlanInputBodyGuide } from './PlanInputBodyGuide/PlanInputBodyGuide'
import { usePlanInputGuideContent } from './PlanInputBodyGuide/UsePlanInputGuideContent'
import { PlanInputBodyHeader } from './PlanInputBodyHeader'

export const PlanInputBodyMobile = React.memo(
  ({
    sizing,
    type,
    children,
    customGuideIntro,
  }: {
    sizing: PlanInputSizing['fixed']
    type: PlanInputType
    customGuideIntro?: ReactNode
    children: {
      content: ReactElement
      error?: ReactElement
      input?: (transitionOut: (onDone: () => void) => void) => ReactElement
    }
  }) => {
    const { params } = useSimulation()
    const { padding } =
      params.dialogPosition !== 'done'
        ? sizing.dialogMode
        : sizing.notDialogMode

    const [showInput, setShowInput] = useState(false)
    const [showError, setShowError] = useState(false)
    const hasInput = children.input !== undefined
    useLayoutEffect(() => {
      if (hasInput) setShowInput(true)
    }, [hasInput])

    useEffect(() => {
      if (!children.error) setShowError(false)
    }, [children.error])
    const guideContent = usePlanInputGuideContent(type)
    return (
      <div className="absolute inset-0 overflow-y-scroll">
        <div
          className=""
          style={{
            ...paddingCSSStyleHorz(newPaddingHorz(padding)),
            paddingTop: `${padding.top}px`,
          }}
        >
          <PlanInputBodyHeader className="mb-6 " type={type} />
          <div className="mb-20">
            {guideContent && (
              <PlanInputBodyGuide
                className="mb-10"
                type={type}
                padding={sizing.cardPadding}
                customIntro={customGuideIntro}
              />
            )}
            {children.content}
          </div>
          {children.error && (
            <button
              className={`fixed rounded-full px-4 py-3 bottom-[25px] right-[25px] bg-pageBG border-2 border-errorFG  text-errorFG font-bold   `}
              style={{ boxShadow: 'rgba(0, 0, 0, 0.35) 0px 5px 10px' }}
              onClick={() => setShowError(true)}
            >
              Warning!
            </button>
          )}
        </div>
        {showInput && (
          <ModalBase>
            {(transitionOut) => (
              <div className="px-2 pb-4">
                {fGet(children.input)((onDone) =>
                  transitionOut(() => {
                    setShowInput(false)
                    onDone()
                  }),
                )}
              </div>
            )}
          </ModalBase>
        )}
        {showError && (
          <ModalBase bg="bg-red-100" onClose={() => setShowError(false)}>
            {() => <div className="  rounded-lg p-2">{children?.error}</div>}
          </ModalBase>
        )}
      </div>
    )
  },
)
