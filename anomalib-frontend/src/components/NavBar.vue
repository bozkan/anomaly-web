<template>
  <header class="bg-blue-500 py-4">
    <nav class="container mx-auto flex items-center justify-between">
      <router-link to="/" class="text-white text-xl font-semibold"
        >Your Logo</router-link
      >
      <div class="space-x-4">
        <router-link to="/training" class="text-white hover:text-gray-300 mr-5"
          >Training</router-link
        >
        <router-link to="/prediction" class="text-white hover:text-gray-300"
          >Prediction</router-link
        >
      </div>
    </nav>
  </header>
</template>

<script>
export default {
  name: "NavBar",
};
</script>

<style scoped>
/* Full-width background color */
header {
  width: 100%;
}

/* Increase the font size and add some padding */
.text-xl {
  font-size: 1.5rem;
}

.py-4 {
  padding-top: 1rem;
  padding-bottom: 1rem;
}

/* Style the navigation links */
.nav-link {
  color: white;
  transition: color 0.3s ease; /* Smooth transition on hover */
}

.nav-link:hover {
  color: #ccc; /* Change color on hover */
}
</style>
