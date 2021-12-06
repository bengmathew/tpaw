import {
  faChevronRight,
  faExclamationCircle
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Disclosure } from '@headlessui/react'
import React, { useState } from 'react'
import Measure, { BoundingRect } from 'react-measure'
import { fGet } from '../../Utils/Utils'
import { useWindowWidth } from '../../Utils/WithWindowSize'

export const CardItem = React.memo(
  ({
    heading,
    subHeading,
    children,
    warn = false,
  }: {
    heading: string
    subHeading: string
    children: React.ReactNode
    warn?: boolean
  }) => {
    const [headingBounds, setHeadingBounds] = useState<BoundingRect | null>(
      null
    )
    const [childBounds, setChildBounds] = useState<BoundingRect | null>(null)
    const windowWidth = useWindowWidth()
    const width = windowWidth <= 500 ? windowWidth - 2 * 8 : 500 - 2 * 8

    return (
      <Disclosure>
        {({open}) => (
          <div
            className={`w-full overflow-hidden  relative`}
            style={{
              width: `${width}px`,
            }}
          >
            <div
              className={`absolute bg-gray-100 rounded-lg z-0
              ${open ? 'bg-opacity-100' : 'bg-opacity-0'}`}
              style={{
                width: `${open ? width : headingBounds?.width ?? 0}px`,
                bottom: '0px',
                top: '0px',
                transition: 'width .25s ease, background-color .25s ease',
              }}
            ></div>
            <Disclosure.Button className="w-full flex rounded-lg relative z-10">
              <Measure
                bounds
                onResize={({bounds}) => {
                  setHeadingBounds(fGet(bounds))
                }}
              >
                {({measureRef}) => (
                  <div
                    className="py-0.5 pl-2 flex items-center gap-x-2 "
                    ref={measureRef}
                  >
                    <FontAwesomeIcon
                      className={` transform ${open ? 'rotate-90' : 'rotate-0'} 
                  transition-transform `}
                      icon={faChevronRight}
                    />
                    <div className=" flex flex-col  items-start font-medium">
                      <h2 className="text-lg text-left">
                        {heading}
                        {warn && (
                          <FontAwesomeIcon
                            className="ml-2 text-red-500"
                            icon={faExclamationCircle}
                          />
                        )}
                      </h2>
                      <h2 className="opacity-40 text-left text-[.90rem]">
                        {subHeading}
                      </h2>
                    </div>
                  </div>
                )}
              </Measure>
            </Disclosure.Button>
            <div
              className="relative z-10 overflow-hidden"
              style={{
                width: `${open ? width : headingBounds?.width ?? 0}px`,
                height: open ? `${childBounds?.height ?? 0}px` : `0px`,
                opacity: open ? `1` : `0`,
                transition: 'height .25s ease, width .25s ease',
              }}
              onTransitionEnd={() => {}}
            >
              <Measure
                bounds
                onResize={({bounds}) => {
                  setChildBounds(fGet(bounds))
                }}
              >
                {({measureRef}) => (
                  <Disclosure.Panel
                    className="absolute"
                    static
                    ref={measureRef}
                    style={{width: width}}
                  >
                    <div className="px-4 py-6 ">{children}</div>
                  </Disclosure.Panel>
                )}
              </Measure>
            </div>
          </div>
        )}
      </Disclosure>
    )
  }
)
