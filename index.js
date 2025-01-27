import express from "express";
import { pipeline } from "@xenova/transformers";

const app = express();

// Middleware to parse JSON request bodies
app.use(express.json());

let embedder;

const loadPipeline = async () => {
  embedder = await pipeline("feature-extraction", "Xenova/nomic-embed-text-v1");
};

// Load the pipeline when the server starts
// Define a route to analyze sentiment
app.post("/getEmbeddings", async (req, res) => {
  try {
    const { data } = req.body;
    if (!data) {
      return res.status(400).json({ error: "No data provided for analysis." });
    }

    const results = await embedder(data, { pooling: "mean", normalize: true });

    res.status(200).json({
      data: Array.from(results.data),
    });
  } catch (error) {
    console.error("Error during analysis:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  await loadPipeline();
  console.log(`Server is running on http://localhost:${PORT}`);
});
