document.getElementById("myForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the text input value
    var textInput = document.getElementById("textInput").value;

    // Send an AJAX request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/process", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log("Response received: " + xhr.responseText);
        }
    };
    xhr.send(JSON.stringify({ text: textInput }));
});
