document.getElementById('categoryForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const data = {};
    for (const [key, value] of formData.entries()) {
        data[key] = value;
    }
    console.log("Form Data Submitted:", data);
    alert("Form data has been logged to the console!");
});
