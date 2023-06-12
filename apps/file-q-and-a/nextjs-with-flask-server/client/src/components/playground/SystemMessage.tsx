import { useOpenAI } from "@/context/OpenAIProvider";
import React from "react";
import TextArea from "../input/TextArea";

type Props = {};

export default function SystemMessage({}: Props) {
  const { updateSystemMessage, systemMessage } = useOpenAI();
  const defaultText = `You are an AI assistant providing helpful advice. 
  You are given the following extracted parts of a long document and a question. 
  Provide a conversational answer based on the context provided.
  Context information is below.
  
  =========
  {files_string}
  =========
  {question}`
  

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
