import { faTwitter } from '@fortawesome/free-brands-svg-icons'
import { faEnvelope } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import clix from 'clsx'
import Image from 'next/image'
import Link from 'next/link'
import React from 'react'
import { ContextMenu2 } from '../../../Common/Modal/ContextMenu2'

export const PlanContactContent = React.memo(
  ({
    className,
    colors,
  }: {
    className?: string
    colors: { bg: string; fg: string }
  }) => {
    return (
      <div
        className={clix(className, 'rounded-full text-lg flex items-stretch ')}
        style={{ backgroundColor: colors.bg, color: colors.fg }}
      >
        <a
          href="https://twitter.com/bmathecon"
          className="pl-4 pr-2 py-2"
          target="_blank"
          rel="noreferrer"
        >
          <FontAwesomeIcon icon={faTwitter} />
        </a>
        <div className="border-l border-gray-400 w-[1px] my-2"></div>
        <ContextMenu2
          className="pl-2 pr-4 py-2 rounded-lg"
          align="right"
        >
          <div className="">Contact</div>
          {({ close, onMenuClose }) => (
            <Menu.Items className="py-2.5 rounded-lg max-w-[350px] ">
              <p className="px-4 p-base">
                This planner is developed by Ben Mathew. If you have any
                questions or comments, you can get in touch with Ben at:
              </p>
              <Menu.Item
                as="a"
                className="context-menu-item items-center cursor-pointer mt-2"
                href="https://www.bogleheads.org/forum/viewtopic.php?t=331368"
                target="_blank"
                rel="noreferrer"
                style={{ display: 'flex' }}
              >
                <div className="w-[25px]">
                  <Image
                    src="/bolgeheads_logo.png"
                    alt="substack icon"
                    width="17"
                    height="15"
                  />
                </div>
                <h2 className="">Bogleheads</h2>
              </Menu.Item>

              <Menu.Item
                as="a"
                href="https://twitter.com/bmathecon"
                target="_blank"
                rel="noreferrer"
                className="context-menu-item "
                onClick={() => {}}
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faTwitter} />
                </span>
                Twitter
              </Menu.Item>

              <Menu.Item
                as="a"
                className="context-menu-item items-center cursor-pointer"
                href="https://substack.com/@benmathew"
                target="_blank"
                rel="noreferrer"
                style={{ display: 'flex' }}
              >
                <div className="w-[25px]">
                  <Image
                    src="/substack_logo.svg"
                    alt="substack icon"
                    width="17"
                    height="15"
                  />
                </div>
                <h2 className="">Substack</h2>
              </Menu.Item>
              <Menu.Item
                as="a"
                className="context-menu-item "
                href="mailto:ben@tpawplanner.com"
                target="_blank"
                rel="noreferrer"
                onClick={() => {}}
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faEnvelope} />
                </span>
                ben@tpawplanner.com
              </Menu.Item>
              <div className="mt-3">
                <Link
                  href={'/about'}
                  target="_blank"
                  className="underline ml-4 pt-2"
                >
                  About Ben
                </Link>
              </div>
            </Menu.Items>
          )}
        </ContextMenu2>
      </div>
    )
  },
)
