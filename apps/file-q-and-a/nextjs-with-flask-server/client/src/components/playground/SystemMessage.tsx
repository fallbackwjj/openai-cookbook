import { useOpenAI } from "@/context/OpenAIProvider";
import React from "react";
import TextArea from "../input/TextArea";

type Props = {};

export default function SystemMessage({}: Props) {
  const { updateSystemMessage, systemMessage } = useOpenAI();
  const defaultText = `Given a question, try to answer it using the content of the file extracts below.

  if you cannot answer, or find a relevant file, just output "I couldn't find the answer to that question in your files.".
  
  If the answer is not contained in the files or if there are no file extracts, respond with "I couldn't find the answer to that question in your files." .
  
  If the question is not actually a question, respond with "That's not a valid question.".
  
  In the cases where you can find the answer, first give the answer. 
  
  Then explain how you found the answer from the source or sources, and use the exact filenames of the source files you mention. 
  
  Do not make up the names of any other files other than those mentioned in the files context.
  
  Give the answer in markdown format，Use the following format:
  Question: <question>Files:<###"filename 1" file text>\n<###"filename 2" file text>...\n\n
  Answer: <answer or "I couldn't find the answer to that question in your files" or "That's not a valid question"> 
  Question: {question}
  Files:\n{files_string}\n
  Answer:`

  return (
    <TextArea
      title="System"
      className="grow"
      placeholder={defaultText}
      value={systemMessage.content}
      onChange={(e) => updateSystemMessage(e.target.value)}
    />
  );
}
