function openTab(event, tabName) {
    var i, tabContents, tabButton;
    tabContents = document.querySelectorAll(".home, .about");
    for (i = 0; i < tabContents.length; i++) {
        tabContents[i].style.display = "none";
    }
    tabButton = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabButton.length; i++) {
        tabButton[i].classList.remove("active");
    }
    document.getElementById(tabName).style.display = "block";
    event.currentTarget.classList.add("active");
}

document.getElementById("Home").style.display = "block";
document.getElementsByClassName("tab-button")[0].classList.add("active");

// Function to handle file input change event
document.getElementById("image").addEventListener("change", function (event) {
    // Get the selected file
    var file = event.target.files[0];

    // Check if a file is selected
    if (file) {
        // Display the selected file name in the h4 element
        document.querySelector("h4").textContent = file.name;

        var reader = new FileReader();

        // Read the image file as a data URL
        reader.readAsDataURL(file);

        // When the file is loaded, set it as the source of the selected image
        reader.onload = function (event) {
            // Set the src attribute of the selected image element to the data URL
            document.getElementById("selected-image").src = event.target.result;
        };
    }
});


document.getElementById("process-image-button").addEventListener("click", function () {
    var inputFile = document.getElementById("image");
    var fileName = inputFile.files[0].name;
    console.log("Selected File Name:", fileName);

    // Send the file name to the backend
    fetch('http://127.0.0.1:5000/process_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file_name: fileName })
    })
    .catch(error => console.error('Error:', error));
});



document.getElementById("reset-image-button").addEventListener("click", function () {
    // Clear the selected image source
    document.getElementById("selected-image").src = "";

    // Clear the file input
    var inputFile = document.getElementById("image");
    inputFile.value = "";

    // Reset the h4 tag content
    document.querySelector("h4").textContent = "";
});

