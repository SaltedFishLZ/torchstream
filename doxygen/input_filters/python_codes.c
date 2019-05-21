#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 4096

#define DOCMARK_SQUOTES ("\'\'\'")
#define DOCMARK_DQUOTES ("\"\"\"")

#define TEXT_STATE_PLAIN        0
#define TEXT_STATE_DOC_SQ       1
#define TEXT_STATE_DOC_SQ_END   2
#define TEXT_STATE_DOC_DQ       3
#define TEXT_STATE_DOC_DQ_END   4


int main(int argc, char const *argv[])
{
    char* filename = argv[1];

    FILE* fin = fopen(filename, "r");

    if (fin != NULL)
    {
        int state = TEXT_STATE_PLAIN;
        char line [MAX_LINE_SIZE];
        while (fgets(line, sizeof(line), fin) != NULL)
        {
            /**
             * state machine transition
             */
            state_transition: {
                if (TEXT_STATE_PLAIN == state)
                {
                /** 
                 * plain text -> docstring with single quotes
                 */
                if (NULL != strstr(line, DOCMARK_SQUOTES))
                {
                    state = TEXT_STATE_DOC_SQ;
                    
                }
                /** 
                 * plain text -> docstring with double quotes
                 */                
                else if (NULL != strstr(line, DOCMARK_DQUOTES))
                {
                    state = TEXT_STATE_DOC_DQ;
                }
                }
                else if (TEXT_STATE_DOC_SQ == state)
                {
                /** 
                 * docstring with single quotes -> 
                 * docstring with single quotes ending state
                 */
                if (NULL != strstr(line, DOCMARK_SQUOTES))
                {
                    state = TEXT_STATE_DOC_SQ_END;
                }
                }
                else if (TEXT_STATE_DOC_DQ == state)
                {
                /** 
                 * docstring with double quotes -> 
                 * docstring with double quotes ending state
                 */                
                if (NULL != strstr(line, DOCMARK_DQUOTES))
                {
                    state = TEXT_STATE_DOC_DQ_END;
                }                
                }
                else if (
                (TEXT_STATE_DOC_SQ_END == state)
                || (TEXT_STATE_DOC_DQ_END == state)
                )
                {
                /** 
                 * docstring with single double quotes  ending state -> 
                 * plain text
                 */                 
                state = TEXT_STATE_PLAIN;
                }
                else
                {
                    state = TEXT_STATE_PLAIN;
                }
            }
            
            /**
             * output stahe
             */
            output: {
                switch (state)
                {
                case TEXT_STATE_PLAIN:
                    printf("%s", line);
                    break;

                default:
                    printf("# DOXYGEN PADDING LINE\n");
                    break;
                }
            }
        }
        fclose (fin);
    }
    else
    {
        perror(filename);
    }




    return 0;
}
