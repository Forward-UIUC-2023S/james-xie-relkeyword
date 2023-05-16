document.addEventListener('DOMContentLoaded', function() {
    chrome.storage.local.get("input_word", function(data) {
        console.log("RECEIVED FROM STORAGE");
        console.log(data);
        var userInput = data.input_word;
        if (userInput) {
            const container2 = document.getElementById("container");
            let html2 = "<h2> Finding Recommendations For " + userInput + "</h2>";
            container2.innerHTML = html2;
        
            console.log(userInput)

            // Initial Request to get the related keywords
            var url = "http://127.0.0.1:8000" + '/get_related_words/?topic='+ encodeURIComponent(userInput) ;
            let results;
            fetch(url).then(response => response.json()).then(data => {
                results = data;
                console.log(results);
                // console.log(results['summary']);
                if (results['word'] == "-1") {
                    const container = document.getElementById("container");
                    let html = "<h2> Could not find Recommendations for Keyword: " + userInput + "</h2>";
                    container.innerHTML = html;
                } else {
                    const container = document.getElementById("container");
                    let html = "<h2> Recommended Keywords for " + userInput + "<h2>";
                    html += "<h2> Semantically Similar Keywords: </h2>";
                    data['semantic_similar'].forEach(message => {
                        html += "<button>" + message[0] + "</button>";
                    });
                    html += "<h2> Semantically Related Keywords: </h2>";
                    data['semantic_related'].forEach(message => {
                        html += "<button>" + message[0] + "</button>";
                    });
                    container.innerHTML = html
                    const buttons = container.querySelectorAll('button');
                    buttons.forEach(button => {
                        button.addEventListener('click', function() {
                            // Make another request to the backend based on the clicked button
                            const clickedButton = button.textContent;
                            const requestData = {
                                word: userInput,
                                related_word: clickedButton
                            };
                            const url2 = "http://127.0.0.1:8000" + '/get_sentence/?word=' + encodeURIComponent(userInput) + '&related_word=' + encodeURIComponent(clickedButton);
                            container.innerHTML = "<h2> retrieving sentence... </h2";
                            fetch(url2).then(response => response.json()).then(data => {
                                // Display the retrieved data on the page
                                container.innerHTML = "<h2>" + clickedButton + " Data: </h2>" + "<p>" + data['sentence'] + "</p>";
                            });
                        });
                    });
                }
        
            })
        }
    });
});




//     var area = document.querySelector('textarea');
//     if (area && area.addEventListener) {
//         area.addEventListener('input', function() {
//           // event handling code for sane browsers
//           console.log("ENTERED");
//           console.log("HDLFKJSDF");
//         }, false);
//       } else if (area && area.attachEvent) {
//         area.attachEvent('onpropertychange', function() {
//           // IE-specific event handling code
//         });
//       }
// });


// chrome.runtime.onMessage.addListener(
//     function(request, sender, sendResponse) {
//         console.log("RECEIVED A MESSAGE");
//         if (request.message === "input_word") {
//             //  To do something
//             console.log("INSIDE INPUT WORD")
            
//             var userInput = request.input_word
//             if (userInput) {

            
//             console.log(userInput)
            
//             var url = "http://127.0.0.1:8000" + '/get_related_words/?topic='+ encodeURIComponent(userInput) ;
//             let results;
//             fetch(url).then(response => response.json()).then(data => {
//                 results = data;
//                 console.log(results);
//                 // console.log(results['summary']);
//                 const container = document.getElementById("container");
//                 let html = "<h2> Semantically Similar Keywords: </h2>";
//                 data['semantic_similar'].forEach(message => {
//                     html += "<button>" + message[0] + "</button>";
//                 });
//                 html += "<h2> Semantically Related Keywords: </h2>";
//                 data['semantic_related'].forEach(message => {
//                     html += "<button>" + message[0] + "</button>";
//                 });
//                 container.innerHTML = html
//                 const buttons = container.querySelectorAll('button');
//                 buttons.forEach(button => {
//                     button.addEventListener('click', function() {
//                         // Make another request to the backend based on the clicked button
//                         const clickedButton = button.textContent;
//                         const requestData = {
//                             word: userInput,
//                             related_word: clickedButton
//                         };
//                         const url2 = "http://127.0.0.1:8000" + '/get_sentence/?word=' + encodeURIComponent(userInput) + '&related_word=' + encodeURIComponent(clickedButton);
//                         fetch(url2).then(response => response.json()).then(data => {
//                             // Display the retrieved data on the page
//                             container.innerHTML = "<h2>" + clickedButton + " Data: </h2>" + "<p>" + data['sentence'] + "</p>";
//                         });
//                     });
//                 });
        
//             })
//         }
//     }
//     }
// );