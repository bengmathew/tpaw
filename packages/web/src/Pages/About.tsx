import { faTwitter } from '@fortawesome/free-brands-svg-icons'
import { faEnvelope } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Image from 'next/image'
import React from 'react'
import { AppPage } from './App/AppPage'

export const About = React.memo(() => {
  return (
    <AppPage className=" pt-header min-h-screen" title={`About - TPAW Planner`}>
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-4 z-0">
          <div className=" ">
            <h1 className="font-bold text-4xl">About</h1>
            <p className="mt-4 p-base">
              TPAW was developed by Ben Mathew on the Bogleheads thread{' '}
              <a
                href="https://www.bogleheads.org/forum/viewtopic.php?t=331368"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                Total portfolio allocation and withdrawal (TPAW)
              </a>
              . Ben has a B.A. in economics from Dartmouth College, and a Ph.D.
              in economics from the University of Chicago. He has taught
              economics at the University of Chicago, and economics and finance
              at Colgate University. You can get in touch with Ben at:
              <a
                className="block py-2  items-center cursor-pointer mt-2"
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
              </a>
              <a
                className="block py-2 cursor-pointer"
                href="https://twitter.com/bmathecon"
                target="_blank"
                rel="noreferrer"
                onClick={() => {}}
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faTwitter} />
                </span>
                Twitter
              </a>
              <a
                className="block py-2  items-center cursor-pointer"
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
              </a>
              <a
                className="block py-2 cursor-pointer"
                href="mailto:ben@tpawplanner.com"
                target="_blank"
                rel="noreferrer"
                onClick={() => {}}
              >
                <span className="inline-block w-[25px]">
                  <FontAwesomeIcon icon={faEnvelope} />
                </span>
                ben@tpawplanner.com
              </a>
            </p>
            <p className="mt-4 p-base">
              The software for this website was written by Ben’s brother, Jacob
              Mathew. Jacob has a B.A. in computer science from Cornell
              University and an M.S. in computer science from UCLA.
            </p>
          </div>
        </div>
      </div>
    </AppPage>
  )
})
