<template>
  <div>
    <input type="file" @change="onFileChange" />
    <button @click="predictAnomaly">Predict</button>
    <div v-if="anomalyRate !== null">Anomaly Rate: {{ anomalyRate }}</div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      file: null,
      anomalyRate: null,
    };
  },
  methods: {
    onFileChange(e) {
      this.file = e.target.files[0];
    },
    async predictAnomaly() {
      const formData = new FormData();
      formData.append("file", this.file);

      try {
        const response = await axios.post(
          "http://localhost:8000/predict",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );
        this.anomalyRate = response.data.anomaly_rate;
      } catch (error) {
        console.error("Error during prediction:", error);
      }
    },
  },
};
</script>
