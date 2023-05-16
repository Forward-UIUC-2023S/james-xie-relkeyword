
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    console.log("Background Recieved an Input Word");
        if (request.message === "input_word") {
            //  To do something
            console.log("INSIDE INPUT WORD")
            
            var userInput = request.input_word
            if (userInput) {

            
                console.log(userInput)
                chrome.storage.local.set({input_word: userInput}, function() {
                    if(chrome.runtime.lastError) {
                      console.error(
                        "Error setting " + key + " to " + JSON.stringify(data) +
                        ": " + chrome.runtime.lastError.message
                      );
                    }
                  });
            }
        }

});