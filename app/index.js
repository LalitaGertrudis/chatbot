import express from "express";
import dotenv from "dotenv";
import pino from "pino";
import pinoHttp from "pino-http";
import bodyParser from "body-parser";
import expressAsyncHandler from "express-async-handler";
import ollama from "./ollama/index.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const log = pino({
    customLevels: {
        metric: 25
    }
});

export const logRequestsMiddleware = pinoHttp({
    logger: log,
});

async function main() {
    console.log("Setting up server");
    const llm = ollama.getOllamaChat();
    const vectorstore = await ollama.initializeVectorWithPDF("./art_22.pdf");

    app.use(logRequestsMiddleware);
    app.use(bodyParser.json());

    app.get("/", (_, res) => {
        res.send("Chatbot express server");
    });

    app.post("/chat", expressAsyncHandler(async (req, res) => {
        const body = req.body;

        if (body.message) {
            const stream = await ollama.queryVectorStore([
                {
                    content: body.message,
                    role: "human",
                },
            ], llm, vectorstore);

            res.json({ success: true, answer: stream });
        } else {
            res.status(400).json({ success: false, message: "Invalid request" });
        }
    }));

    app.listen(port, () => {
        console.log(`Server is running on port ${port}`);
    });
}

main();
