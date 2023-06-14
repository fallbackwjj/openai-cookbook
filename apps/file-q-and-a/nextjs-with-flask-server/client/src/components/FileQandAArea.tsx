import React, { memo, useCallback, useRef, useState } from "react";
import { Transition } from "@headlessui/react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

import FileViewerList from "./FileViewerList";
import LoadingText from "./LoadingText";
import { isFileNameInString } from "../services/utils";
import { FileChunk, FileLite } from "../types/file";
import { SERVER_ADDRESS } from "../types/constants";

import PlaygroundMessages from "@/components/playground/PlaygroundMessages";
import ConfigSidebar from "@/components/playground/ConfigSidebar";
import PlaygroundHeader from "@/components/playground/PlaygroundHeader";
import SystemMessage from "@/components/playground/SystemMessage";
import PlaygroundConversations from "@/components/playground/conversations/PlaygroundConversations";
import PlaygroundProvider from "@/context/PlaygroundProvider";

type FileQandAAreaProps = {
  files: FileLite[];
};

function FileQandAArea(props: FileQandAAreaProps) {
  const searchBarRef = useRef(null);
  const [answerError, setAnswerError] = useState("");
  const [searchResultsLoading, setSearchResultsLoading] =
    useState<boolean>(false);
  const [answer, setAnswer] = useState("");

  const handleSearch = useCallback(async () => {
    if (searchResultsLoading) {
      return;
    }

    const question = (searchBarRef?.current as any)?.value ?? "";
    setAnswer("");

    if (!question) {
      setAnswerError("Please ask a question.");
      return;
    }
    if (props.files.length === 0) {
      setAnswerError("Please upload files before asking a question.");
      return;
    }

    setSearchResultsLoading(true);
    setAnswerError("");

    let results: FileChunk[] = [];
    var baseUrl = `${SERVER_ADDRESS}/answer_question`;
    // if (`${process.env.NODE_ENV}` != "development") {
    //   baseUrl = "/backend/answer_question";
    // }
    try {
      const answerResponse = await axios.post(
        baseUrl,
        {
          question,
        },
        {
          withCredentials: true
        }
      );

      if (answerResponse.status === 200) {
        setAnswer(answerResponse.data.answer);
      } else {
        setAnswerError("Sorry, something went wrong!");
      }
    } catch (err: any) {
      setAnswerError("Sorry, something went wrong!");
    }

    setSearchResultsLoading(false);
  }, [props.files, searchResultsLoading]);

  const handleEnterInSearchBar = useCallback(
    async (event: React.SyntheticEvent) => {
      if ((event as any).key === "Enter") {
        await handleSearch();
      }
    },
    [handleSearch]
  );

  return (
    <div className="space-y-4 text-gray-800">
      <div className="mt-2">
        Ask a question based on the content of your files:
      </div>
      <React.Fragment>
          <main className="relative flex max-h-screen flex-col">
            <PlaygroundProvider>
              <PlaygroundHeader />
              <div className="flex h-[calc(100vh-60px)] max-h-[calc(100vh-60px)] grow flex-row">
                <div className="flex grow flex-col items-stretch md:flex-row">
                  <PlaygroundConversations />
                  <div className="flex grow">
                    <SystemMessage />
                  </div>
                  <div className="flex grow basis-7/12 overflow-hidden">
                    <PlaygroundMessages files = {props.files} />
                  </div>
                </div>
                {/* <ConfigSidebar /> */}
              </div>
            </PlaygroundProvider>
          </main>
      </React.Fragment>
{/* 
      <div className="space-y-2">
        <input
          className="border rounded border-gray-200 w-full py-1 px-2"
          placeholder="e.g. What were the key takeaways from the Q1 planning meeting?"
          name="search"
          ref={searchBarRef}
          onKeyDown={handleEnterInSearchBar}
        />

        <div
          className="rounded-md bg-gray-50 py-1 px-4 w-max text-gray-500 hover:bg-gray-100 border border-gray-100 shadow cursor-pointer"
          onClick={handleSearch}
        >
          {searchResultsLoading ? (
            <LoadingText text="Answering question..." />
          ) : (
            "Ask question"
          )}
        </div>
      </div>
      <div className="">
        {answerError && <div className="text-red-500">{answerError}</div>}
        <Transition
          show={answer !== ""}
          enter="transition duration-600 ease-out"
          enterFrom="transform opacity-0"
          enterTo="transform opacity-100"
          leave="transition duration-125 ease-out"
          leaveFrom="transform opacity-100"
          leaveTo="transform opacity-0"
          className="mb-8"
        >
          {answer && (
            <div className="">
              <ReactMarkdown className="prose" linkTarget="_blank" escapeHtml="true">
                {answer}
              </ReactMarkdown>
            </div>
          )}

          <Transition
            show={
              props.files.filter((file) =>
                isFileNameInString(file.name, answer)
              ).length > 0
            }
            enter="transition duration-600 ease-out"
            enterFrom="transform opacity-0"
            enterTo="transform opacity-100"
            leave="transition duration-125 ease-out"
            leaveFrom="transform opacity-100"
            leaveTo="transform opacity-0"
            className="mb-8"
          >
            <FileViewerList
              files={props.files.filter((file) =>
                isFileNameInString(file.name, answer)
              )}
              title="Sources"
              listExpanded={true}
            />
          </Transition>
        </Transition>
      </div>
       */}
    </div>
  );
}

export default memo(FileQandAArea);
