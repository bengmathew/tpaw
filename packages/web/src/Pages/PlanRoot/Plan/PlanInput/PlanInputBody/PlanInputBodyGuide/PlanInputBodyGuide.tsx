import React, {ReactNode, useState} from 'react'
import {Padding, paddingCSSStyle} from '../../../../../../Utils/Geometry'
import {ModalBase} from '../../../../../Common/Modal/ModalBase'
import {PlanInputType} from '../../Helpers/PlanInputType'
import {PlanInputBodyGuideContent} from './PlanInputBodyGuideContent'
import {usePlanInputGuideContent} from './UsePlanInputGuideContent'

export const PlanInputBodyGuide = React.memo(
  ({
    className = '',
    padding,
    type,
    customIntro = `This is an advanced section. Here is a guide to help you understand how
    to use it.`,
  }: {
    className?: string
    padding: Padding
    type: PlanInputType
    customIntro?: ReactNode
  }) => {
    const [showGuide, setShowGuide] = useState(false)

    const guideContent = usePlanInputGuideContent(type)
    return (
      <div
        className={`${className} border border-gray-400 rounded-2xl p-base`}
        style={{...paddingCSSStyle(padding)}}
      >{customIntro}{' '}
        <button className="underline" onClick={() => setShowGuide(true)}>
          Open Guide
        </button>
        {showGuide && guideContent && (
          <ModalBase onClose={() => setShowGuide(false)}>
            {transitionOut => (
              <div className="px-2 py-4">
                <h2 className="text-2xl mb-4 w-full font-bold text-center">
                  Guide
                </h2>
                <PlanInputBodyGuideContent content={guideContent} />
              </div>
            )}
          </ModalBase>
        )}
      </div>
    )
  }
)
