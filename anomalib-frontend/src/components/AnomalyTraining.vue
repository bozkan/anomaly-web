<template>
  <div>
    <input type="file" @change="onFileChange" />
    <button @click="trainModel">Train</button>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      file: null,
    };
  },
  methods: {
    onFileChange(e) {
      this.file = e.target.files[0];
    },
    async trainModel() {
      const formData = new FormData();
      formData.append("file", this.file);

      try {
        const response = await axios.post(
          "http://localhost:8000/train",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );
        alert(response.data.message);
      } catch (error) {
        console.error("Error during training:", error);
      }
    },
  },
};
</script>
